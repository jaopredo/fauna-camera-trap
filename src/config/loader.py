"""Carregamento de configuracao a partir de YAML, com sobrescritas de linha de comando."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.config.schema import ConfigError, ExperimentConfig


def load_config(
    path: Path | str, *, overrides: Sequence[str] | None = None
) -> ExperimentConfig:
    """Le o YAML em ``path`` e devolve uma configuracao nova.

    ``overrides`` aceita pares ``chave.aninhada=valor`` (ex.: ``seed=1``,
    ``data.budget=20``), o que permite a Parte 2 varrer seeds e orcamentos sem
    criar um YAML por combinacao.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"configuracao nao encontrada: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: o YAML precisa ser um mapeamento na raiz")
    merged = _apply_overrides(dict(raw), overrides or ())
    return ExperimentConfig.from_raw(merged)


def _apply_overrides(raw: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ConfigError(f"sobrescrita invalida: {item!r}. Use chave=valor")
        target = raw
        parts = key.split(".")
        for part in parts[:-1]:
            nested = target.setdefault(part, {})
            if not isinstance(nested, dict):
                raise ConfigError(
                    f"sobrescrita invalida: {key!r} atravessa um valor simples"
                )
            target = nested
        # o valor vem do YAML para que "20" vire int e "true" vire bool,
        # mantendo a mesma conversao do arquivo
        target[parts[-1]] = yaml.safe_load(value)
    return raw
