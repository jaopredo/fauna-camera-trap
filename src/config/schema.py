"""Configuracao de um experimento como dataclasses congeladas.

Sem Singleton e sem estado global: ``load_config`` devolve um objeto novo a cada
chamada, e os ``scripts/`` o injetam nos componentes. Isso importa na Parte 2, que
roda 3 seeds x 3 orcamentos — com uma instancia global compartilhada, a segunda
configuracao carregada seria silenciosamente a primeira.

Toda secao recusa chave desconhecida. Descobrir um ``lr`` escrito como ``Ir`` depois
de meia hora de treino e caro; falhar no carregamento e gratis.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from src.data.splits import DEFAULT_BUDGET, validate_budget
from src.errors import AppError


class ConfigError(AppError, ValueError):
    """Configuracao malformada: chave desconhecida, ausente ou de tipo errado."""


T = TypeVar("T", bound="_Section")


def _reject_unknown(data: Mapping[str, Any], known: set[str], where: str) -> None:
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"{where}: chaves desconhecidas {sorted(unknown)}. Aceitas: {sorted(known)}"
        )


def _normalize_seeds(seeds: Any) -> tuple[int, ...]:
    """Aceita ``0`` ou ``[0, 42, 123]`` e devolve sempre uma tupla validada.

    Um inteiro solto vira tupla de um elemento: escrever ``seeds: 0`` num YAML e
    natural e nao deveria estourar com erro de iteracao sobre int.
    """
    if isinstance(seeds, bool):
        raise ConfigError(f"seeds precisa ser inteiro ou lista, veio {seeds!r}")
    if isinstance(seeds, int):
        seeds = (seeds,)
    try:
        normalizadas = tuple(int(s) for s in seeds)
    except (TypeError, ValueError):
        raise ConfigError(
            f"seeds precisa ser um inteiro ou uma lista de inteiros, veio {seeds!r}"
        ) from None
    if not normalizadas:
        raise ConfigError("seeds: declare ao menos uma seed")
    repetidas = sorted({s for s in normalizadas if normalizadas.count(s) > 1})
    if repetidas:
        raise ConfigError(
            f"seeds repetidas: {repetidas}. Cada seed e uma execucao; repetir "
            f"so sobrescreveria o mesmo diretorio."
        )
    return normalizadas


@dataclass(frozen=True)
class _Section:
    """Base das secoes: construcao a partir de dicionario com chaves validadas."""

    @classmethod
    def from_mapping(cls: type[T], data: Mapping[str, Any] | None, *, where: str) -> T:
        data = dict(data or {})
        _reject_unknown(data, {f.name for f in fields(cls)}, where)
        missing = [
            f.name
            for f in fields(cls)
            if f.name not in data
            and f.default is MISSING
            and f.default_factory is MISSING
        ]
        if missing:
            raise ConfigError(
                f"{where}: chaves obrigatorias ausentes {sorted(missing)}"
            )
        return cls(**data)


@dataclass(frozen=True)
class DataConfig(_Section):
    """Onde estao os dados e como as imagens chegam ao modelo."""

    root: Path = Path("data")
    budget: int = DEFAULT_BUDGET
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    #: nome da politica de aumentacao registrada em ``src.data.augment``
    augment: str = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        validate_budget(self.budget)
        for name in ("image_size", "batch_size"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"data.{name} precisa ser positivo")
        if self.num_workers < 0:
            raise ConfigError("data.num_workers nao pode ser negativo")


@dataclass(frozen=True)
class ModelConfig(_Section):
    """Qual backbone construir. ``params`` sao os argumentos proprios dele."""

    name: str
    pretrained: bool = False
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyConfig(_Section):
    """Qual metodo de treino usar. ``params`` sao os hiperparametros dele.

    E aqui que as trilhas divergem (supervisionado, SimCLR/SwAV, FixMatch,
    professor-aluno) sem que o laco de treino mude.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimConfig(_Section):
    """Otimizacao e duracao do treino."""

    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    #: metrica de validacao que define o melhor checkpoint — escolhida so no val
    monitor: str = "macro_f1"
    #: de quantas em quantas epocas gravar o estado retomavel
    save_every: int = 1

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ConfigError("optim.epochs precisa ser positivo")
        if self.lr <= 0:
            raise ConfigError("optim.lr precisa ser positivo")
        if self.save_every <= 0:
            raise ConfigError("optim.save_every precisa ser positivo")


@dataclass(frozen=True)
class ExperimentConfig(_Section):
    """Um experimento completo e reproduzivel."""

    name: str
    model: ModelConfig
    strategy: StrategyConfig
    #: Seeds que o experimento declara. Uma execucao e sempre UMA seed; o
    #: experimento e o conjunto delas. O enunciado pede 3 seeds no orcamento de 5
    #: (media +- desvio) e uma seed nos de 1 e 20 — e cada YAML declara isso, em
    #: vez de a regra ficar escrita num laco de shell.
    seeds: tuple[int, ...] = (0,)
    output_dir: Path = Path("outputs")
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.name:
            raise ConfigError("name e obrigatorio: e ele que nomeia o diretorio da run")
        object.__setattr__(self, "seeds", _normalize_seeds(self.seeds))

    def run_name(self, seed: int) -> str:
        """Nome do diretorio de uma execucao deste experimento, para uma seed.

        Fonte unica da convencao ``<nome>_b<orcamento>_s<seed>``: o treino grava
        com ela e os relatorios procuram por ela. Duas f-strings separadas
        divergiriam no dia em que a convencao mudasse.
        """
        if seed not in self.seeds:
            raise ConfigError(
                f"seed {seed} nao esta declarada em seeds={list(self.seeds)}"
            )
        return f"{self.name}_b{self.data.budget}_s{seed}"

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "ExperimentConfig":
        """Monta a configuracao a partir do YAML cru, validando cada secao."""
        raw = dict(raw)
        if "seed" in raw:
            raise ConfigError(
                "use `seeds` (lista) em vez de `seed`: o experimento declara o "
                "conjunto de seeds e cada execucao roda uma delas. "
                "Ex.: seeds: [0, 42, 123]"
            )
        _reject_unknown(raw, {f.name for f in fields(cls)}, "raiz")
        sections = {
            "data": (DataConfig, raw.pop("data", None)),
            "model": (ModelConfig, raw.pop("model", None)),
            "strategy": (StrategyConfig, raw.pop("strategy", None)),
            "optim": (OptimConfig, raw.pop("optim", None)),
        }
        built = {
            key: section.from_mapping(value, where=key)
            for key, (section, value) in sections.items()
        }
        return cls.from_mapping({**raw, **built}, where="raiz")

    def as_dict(self) -> dict[str, Any]:
        """Forma serializavel, gravada no manifesto de cada execucao."""
        return {
            "name": self.name,
            "seeds": list(self.seeds),
            "output_dir": str(self.output_dir),
            "data": {
                **{k: v for k, v in vars(self.data).items() if k != "root"},
                "root": str(self.data.root),
            },
            "model": {**vars(self.model), "params": dict(self.model.params)},
            "strategy": {**vars(self.strategy), "params": dict(self.strategy.params)},
            "optim": dict(vars(self.optim)),
        }
