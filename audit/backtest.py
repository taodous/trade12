"""Backtest de la stratégie, net de frais et de slippage.

Le backtest in-sample ne peut pas lire l'out-of-sample : les données sont coupées à
`in_sample_end` avant tout calcul. L'out-of-sample n'est évalué que par
`evaluate_out_of_sample`, appelé sur demande explicite, et chaque appel est journalisé. Un signal
calculé sur la clôture de t s'exécute à l'ouverture de t+1.

Limites du modèle : ni le kill switch ni les contrôles de `bot.risk` ne sont appliqués, le cash ne
rapporte rien, et le prix d'exécution ne tient compte ni du volume ni de l'écart réel à l'ouverture.
Sans politique de rééquilibrage, les positions évoluent librement : quand les gagnantes ont
absorbé le cash, un achat est réduit ou ignoré plutôt que financé par de la marge. Avec un
rééquilibrage mensuel, tous les poids sont remis à leur cible à la première séance du mois, même
pour un petit écart : la bande du paper trading n'est pas simulée, ce qui compte plus de frais
que dans la réalité.
"""

import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
import vectorbt as vbt
from pydantic import BaseModel, ConfigDict

from audit.metrics import (
    Criterion,
    KillSwitchReplay,
    PerformanceMetrics,
    TradeStatistics,
    compute_metrics,
    compute_trade_statistics,
    evaluate_targets,
    replay_kill_switch,
    suspicious_results,
)
from audit.registry import record_variant, variant_id
from bot.config import (
    PROJECT_ROOT,
    BacktestConfig,
    ConfigError,
    MovingAverageSettings,
    RebalanceConfig,
    RiskTargetedSettings,
    StrategyConfig,
    load_strategy_config,
)
from bot.data.store import BarStore, StoreError
from bot.logs import configure_logging
from bot.strategy.base import WeightStrategy
from bot.strategy.factory import STRATEGY_LABELS, build_strategy, rebalance_policy

logger = structlog.get_logger()


class BacktestError(ValueError):
    """Données ou paramètres incompatibles avec le backtest."""


class BacktestReport(BaseModel):
    """Résultat complet d'un backtest in-sample."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    sample: str
    variant: str
    start: date
    end: date
    costs: dict[str, float]
    strategy: PerformanceMetrics
    benchmark: PerformanceMetrics
    trades: TradeStatistics
    open_trades: int
    orders: int
    mean_exposure: float
    max_exposure: float
    kill_switch_replay: KillSwitchReplay
    criteria: list[Criterion]
    suspicious: list[str]


@dataclass(frozen=True)
class BacktestRun:
    """Rendements et trades d'une simulation, sur la période d'évaluation."""

    returns: pd.Series
    closed_trade_pnl: np.ndarray
    open_trades: int
    orders: int
    mean_exposure: float
    max_exposure: float


# Tolérance sur les poids d'une stratégie, pour les arrondis du calcul flottant.
_WEIGHT_TOLERANCE = 1e-9


def executed_targets(strategy: WeightStrategy, closes: pd.DataFrame) -> pd.DataFrame:
    """Retourne le poids à tenir à chaque ouverture : celui visé après la clôture précédente.

    Lève `BacktestError` si la stratégie vise un poids manquant, négatif, au-delà de son plafond
    ou une exposition totale au-delà de 1.
    """
    targets = strategy.target_weights(closes)
    if not targets.index.equals(closes.index) or list(targets.columns) != list(closes.columns):
        raise BacktestError("Les poids doivent être alignés sur les prix")
    values = targets.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise BacktestError("Poids manquant ou négatif")
    if (values > strategy.max_weight + _WEIGHT_TOLERANCE).any():
        raise BacktestError("Poids au-delà du plafond de la stratégie")
    if (values.sum(axis=1) > 1 + _WEIGHT_TOLERANCE).any():
        raise BacktestError("Exposition visée supérieure au capital")
    return targets.shift(1).fillna(0.0)


def _first_sessions_of_month(index: pd.Index) -> np.ndarray:
    dates = pd.DatetimeIndex(index)
    months = np.asarray(dates.year * 12 + dates.month)
    return np.concatenate([[True], months[1:] != months[:-1]])


def order_targets(
    executed: pd.DataFrame, start: int, rebalance: RebalanceConfig | None
) -> pd.DataFrame:
    """Retourne les poids visés par les ordres, vides quand aucun ordre n'est passé.

    Sans rééquilibrage, un ordre n'est passé que si le poids à tenir change. Avec un rééquilibrage
    mensuel, tous les symboles sont remis à leur poids à la première séance de chaque mois. Aucun
    ordre n'est passé avant la séance `start`, où le portefeuille, vide, prend les poids du moment.
    """
    if not 0 <= start < len(executed):
        raise BacktestError(f"Séance de départ {start} hors des {len(executed)} séances")
    if rebalance is None:
        trade = executed.ne(executed.shift(1, fill_value=0.0)).to_numpy(copy=True)
    else:
        first = _first_sessions_of_month(executed.index)
        trade = np.repeat(first[:, None], executed.shape[1], axis=1)
    if start > 0:
        trade[:start] = False
        trade[start] = True
    return executed.where(trade)


def _simulate(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    config: BacktestConfig,
    evaluation_start: int,
) -> BacktestRun:
    # Le poids est valorisé au prix d'ouverture : la clôture du jour n'est pas encore connue.
    portfolio = vbt.Portfolio.from_orders(
        close=closes,
        size=weights,
        size_type="targetpercent",
        price=opens,
        val_price=opens,
        init_cash=config.initial_capital,
        fees=config.fees_pct,
        slippage=config.slippage_pct,
        cash_sharing=True,
        group_by=True,
        call_seq="auto",
        freq="1D",
    )
    trades = portfolio.trades
    exposure = (portfolio.asset_value() / portfolio.value()).iloc[evaluation_start:]
    return BacktestRun(
        returns=portfolio.returns().iloc[evaluation_start:],
        closed_trade_pnl=trades.closed.records_readable["PnL"].to_numpy(),
        open_trades=int(trades.open.count()),
        orders=int(portfolio.orders.count()),
        mean_exposure=float(exposure.mean()),
        max_exposure=float(exposure.max()),
    )


def _validate_prices(opens: pd.DataFrame, closes: pd.DataFrame, warm_up: int) -> None:
    if opens.isna().any().any() or closes.isna().any().any():
        raise BacktestError("Prix manquants : un ordre serait ignoré sans avertissement")
    if not opens.index.equals(closes.index) or list(opens.columns) != list(closes.columns):
        raise BacktestError("Les ouvertures et les clôtures doivent être alignées")
    if len(closes) <= warm_up:
        raise BacktestError(f"Moins de {warm_up + 1} séances disponibles")


def run_strategy(
    strategy: WeightStrategy,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    start: int,
    config: BacktestConfig,
    rebalance: RebalanceConfig | None,
) -> BacktestRun:
    """Exécute une stratégie à partir de la séance `start`, portefeuille vide à cette séance."""
    _validate_prices(opens, closes, start)
    orders = order_targets(executed_targets(strategy, closes), start, rebalance)
    return _simulate(opens, closes, orders, config, start)


def run_weights(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    evaluation_start: int,
    config: BacktestConfig,
) -> BacktestRun:
    """Exécute des poids visés déjà calculés, évalués à partir de `evaluation_start`."""
    _validate_prices(opens, closes, evaluation_start)
    if not weights.index.equals(closes.index) or list(weights.columns) != list(closes.columns):
        raise BacktestError("Les poids doivent être alignés sur les prix")
    return _simulate(opens, closes, weights, config, evaluation_start)


def run_buy_and_hold(
    opens: pd.Series, closes: pd.Series, warm_up: int, config: BacktestConfig
) -> BacktestRun:
    """Exécute l'achat du benchmark, aux mêmes coûts et à la même date que la stratégie."""
    opens_frame = opens.to_frame()
    closes_frame = closes.to_frame()
    _validate_prices(opens_frame, closes_frame, warm_up)
    weights = pd.DataFrame(np.nan, index=closes_frame.index, columns=closes_frame.columns)
    weights.iloc[warm_up, 0] = 1.0
    return _simulate(opens_frame, closes_frame, weights, config, warm_up)


def load_in_sample(store: BarStore, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge les ouvertures et les clôtures, coupées à la fin de l'in-sample."""
    end = pd.Timestamp(config.backtest.in_sample_end)
    opens = store.load_column(config.data.symbols, "open").loc[:end]
    closes = store.load_column(config.data.symbols, "close").loc[:end]
    return opens, closes


def configured_strategy(
    config: StrategyConfig, settings: MovingAverageSettings | RiskTargetedSettings | None = None
) -> WeightStrategy:
    """Construit la stratégie configurée, en vérifiant la limite par position du module risque."""
    strategy = build_strategy(settings or config.strategy, config.data.symbols)
    if strategy.max_weight > config.risk.max_position_pct:
        raise BacktestError(
            f"Poids de {strategy.max_weight:.4f} supérieur à la limite par position"
        )
    return strategy


def variant_definition(
    config: StrategyConfig, settings: MovingAverageSettings | RiskTargetedSettings | None = None
) -> dict[str, Any]:
    """Retourne ce qui définit une variante au registre : stratégie, réglages, univers et coûts.

    La forme reste celle de l'étape 7 pour la stratégie de référence, afin que ses variantes déjà
    inscrites gardent leur identifiant.
    """
    chosen = settings or config.strategy
    definition: dict[str, Any] = {
        "strategy": STRATEGY_LABELS[chosen.name],
        "params": chosen.params.model_dump(),
        "symbols": config.data.symbols,
        "start_date": config.data.start_date,
        "in_sample_end": config.backtest.in_sample_end,
        "fees_pct": config.backtest.fees_pct,
        "slippage_pct": config.backtest.slippage_pct,
    }
    rebalance = rebalance_policy(chosen)
    if rebalance is not None:
        definition["rebalance"] = rebalance.model_dump()
    return definition


@dataclass(frozen=True)
class Evaluation:
    """Rapport d'un backtest, avec les simulations et les prix qui l'ont produit."""

    report: BacktestReport
    run: BacktestRun
    benchmark_run: BacktestRun
    opens: pd.DataFrame
    closes: pd.DataFrame


def _build_report(
    sample: str,
    config: StrategyConfig,
    definition: dict[str, Any],
    run: BacktestRun,
    benchmark_run: BacktestRun,
) -> BacktestReport:
    metrics = compute_metrics(run.returns)
    benchmark = compute_metrics(benchmark_run.returns)
    return BacktestReport(
        sample=sample,
        variant=variant_id(definition),
        start=pd.Timestamp(run.returns.index[0]).date(),
        end=pd.Timestamp(run.returns.index[-1]).date(),
        costs={
            "fees_pct": config.backtest.fees_pct,
            "slippage_pct": config.backtest.slippage_pct,
        },
        strategy=metrics,
        benchmark=benchmark,
        trades=compute_trade_statistics(run.closed_trade_pnl),
        open_trades=run.open_trades,
        orders=run.orders,
        mean_exposure=run.mean_exposure,
        max_exposure=run.max_exposure,
        kill_switch_replay=replay_kill_switch(run.returns, config.risk),
        criteria=evaluate_targets(metrics, benchmark, config.targets),
        suspicious=suspicious_results(metrics, config.targets),
    )


def _write_report(path: Path, report: BacktestReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def evaluate_in_sample(
    config: StrategyConfig, root: Path, now: datetime, report_path: Path | None = None
) -> Evaluation:
    """Lance le backtest in-sample, écrit son rapport et l'inscrit au registre des variantes.

    Le rapport va dans `backtest.report_file`, sauf si `report_path` est fourni.
    """
    backtest = config.backtest
    strategy = configured_strategy(config)
    opens, closes = load_in_sample(BarStore(root / config.data.directory), config)
    warm_up = strategy.warm_up
    run = run_strategy(
        strategy, opens, closes, warm_up, backtest, rebalance_policy(config.strategy)
    )
    benchmark_run = run_buy_and_hold(
        opens[backtest.benchmark], closes[backtest.benchmark], warm_up, backtest
    )

    definition = variant_definition(config)
    report = _build_report("in_sample", config, definition, run, benchmark_run)
    _write_report(report_path or root / backtest.report_file, report)
    record_variant(
        root / backtest.registry_file,
        definition,
        {
            "annual_return": report.strategy.annual_return,
            "sharpe": report.strategy.sharpe,
            "max_drawdown": report.strategy.max_drawdown,
        },
        now,
    )
    return Evaluation(report, run, benchmark_run, opens, closes)


def run_in_sample(config: StrategyConfig, root: Path, now: datetime) -> BacktestReport:
    """Lance le backtest in-sample et retourne son rapport."""
    return evaluate_in_sample(config, root, now).report


def log_out_of_sample_access(path: Path, definition: dict[str, Any], now: datetime) -> None:
    """Inscrit une ouverture de l'out-of-sample, avant tout calcul.

    Chaque ouverture compte : regarder l'out-of-sample puis modifier la stratégie le contamine.
    """
    entry = {"opened_at": now.isoformat(), "variant": variant_id(definition)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry, sort_keys=True) + "\n")


def evaluate_out_of_sample(config: StrategyConfig, root: Path, now: datetime) -> Evaluation:
    """Évalue la stratégie sur les séances qui suivent `in_sample_end`.

    Les signaux sont calculés sur tout l'historique, ce qui ne fait entrer aucune donnée future :
    chaque moyenne n'utilise que les clôtures passées. Le portefeuille part vide à la première
    séance out-of-sample, et le benchmark est acheté à la même séance. Le résultat n'est pas
    inscrit au registre, qui ne doit contenir que des résultats in-sample.
    """
    backtest = config.backtest
    strategy = configured_strategy(config)
    definition = variant_definition(config)
    log_out_of_sample_access(root / config.audit.out_of_sample_log, definition, now)

    store = BarStore(root / config.data.directory)
    opens = store.load_column(config.data.symbols, "open")
    closes = store.load_column(config.data.symbols, "close")
    after_split = closes.index > pd.Timestamp(backtest.in_sample_end)
    if not after_split.any():
        raise BacktestError("Aucune séance après la fin de l'in-sample")
    start = int(np.argmax(after_split))
    if start < strategy.warm_up:
        raise BacktestError("Historique insuffisant avant l'out-of-sample pour le signal")

    run = run_strategy(strategy, opens, closes, start, backtest, rebalance_policy(config.strategy))
    benchmark_run = run_buy_and_hold(
        opens[backtest.benchmark], closes[backtest.benchmark], start, backtest
    )
    report = _build_report("out_of_sample", config, definition, run, benchmark_run)
    _write_report(root / config.audit.report_dir / "backtest_out_of_sample.json", report)
    return Evaluation(report, run, benchmark_run, opens, closes)


def main() -> int:
    """Lance le backtest in-sample et retourne le code de sortie du processus."""
    configure_logging()
    try:
        report = run_in_sample(load_strategy_config(), PROJECT_ROOT, datetime.now(UTC))
    except (ConfigError, StoreError, BacktestError) as exc:
        logger.error("backtest_failed", reason=str(exc))
        return 1
    logger.info("backtest_done", **report.model_dump(mode="json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
