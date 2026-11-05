"""Comparaison in-sample des variantes pré-enregistrées et règle de sélection.

Chaque variante de `candidates` est backtestée in-sample et inscrite au registre. La règle de
sélection a été fixée avant le premier résultat : le meilleur Sharpe in-sample parmi les
variantes dont le drawdown in-sample ne dépasse pas le seuil de rejet. Si aucune ne passe ce
filtre, la meilleure au Sharpe est retenue quand même et signalée : l'audit la rejettera.

La variante retenue doit ensuite être recopiée dans la section `strategy` de la configuration,
pour être auditée complètement. L'out-of-sample n'est pas ouvert ici.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from audit.backtest import BacktestError, evaluate_in_sample, variant_definition
from audit.registry import variant_id
from bot.config import PROJECT_ROOT, ConfigError, StrategyConfig, load_strategy_config
from bot.data.store import StoreError
from bot.logs import configure_logging

logger = structlog.get_logger()

CANDIDATES_DIR = "candidates"
CANDIDATES_REPORT = "candidates.json"


class CandidateResult(BaseModel):
    """Résultat in-sample d'une variante."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    index: int
    variant: str
    settings: dict[str, object]
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    positive_months: float
    eligible: bool


class CandidatesReport(BaseModel):
    """Comparaison des variantes et variante retenue."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    generated_at: datetime
    results: list[CandidateResult]
    selected: int
    none_eligible: bool


def select_candidate(results: list[CandidateResult]) -> tuple[int, bool]:
    """Applique la règle de sélection et indique si aucune variante n'était éligible.

    Un Sharpe indéfini est classé en dernier. À égalité, la première variante l'emporte.
    """
    if not results:
        raise ValueError("Aucune variante à comparer")
    eligible = [item for item in results if item.eligible]
    pool = eligible or results

    def rank(item: CandidateResult) -> float:
        return item.sharpe if item.sharpe == item.sharpe else float("-inf")

    best = max(pool, key=rank)
    return best.index, not eligible


def compare_candidates(config: StrategyConfig, root: Path, now: datetime) -> CandidatesReport:
    """Backtest in-sample chaque variante, l'inscrit au registre et applique la règle."""
    report_dir = root / config.audit.report_dir
    results = []
    for index, settings in enumerate(config.candidates):
        variant_config = config.model_copy(update={"strategy": settings})
        identifier = variant_id(variant_definition(variant_config))
        path = report_dir / CANDIDATES_DIR / f"{identifier}.json"
        metrics = evaluate_in_sample(variant_config, root, now, path).report.strategy
        results.append(
            CandidateResult(
                index=index,
                variant=identifier,
                settings=settings.model_dump(mode="json"),
                annual_return=metrics.annual_return,
                annual_volatility=metrics.annual_volatility,
                sharpe=metrics.sharpe,
                max_drawdown=metrics.max_drawdown,
                calmar=metrics.calmar,
                positive_months=metrics.positive_months,
                eligible=abs(metrics.max_drawdown) <= config.targets.max_drawdown_reject,
            )
        )
        logger.info(
            "candidate_done",
            index=index,
            variant=identifier,
            sharpe=round(metrics.sharpe, 3),
            max_drawdown=round(metrics.max_drawdown, 3),
        )

    selected, none_eligible = select_candidate(results)
    report = CandidatesReport(
        generated_at=now, results=results, selected=selected, none_eligible=none_eligible
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / CANDIDATES_REPORT).write_text(report.model_dump_json(indent=2), "utf-8")
    return report


def main() -> int:
    """Compare les variantes et retourne le code de sortie du processus."""
    configure_logging()
    try:
        report = compare_candidates(load_strategy_config(), PROJECT_ROOT, datetime.now(UTC))
    except (ConfigError, StoreError, BacktestError, ValueError) as exc:
        logger.error("candidates_failed", reason=str(exc))
        return 1
    logger.info(
        "candidates_done",
        selected=report.selected,
        variant=report.results[report.selected].variant,
        none_eligible=report.none_eligible,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
