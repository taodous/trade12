"""Backtest in-sample de la stratégie, net de frais et de slippage.

Ce module ne peut pas lire l'out-of-sample : les données sont coupées à `in_sample_end` avant tout
calcul. Un signal calculé sur la clôture de t s'exécute à l'ouverture de t+1.

Limites du modèle : ni le kill switch ni les contrôles de `bot.risk` ne sont appliqués, le cash ne
rapporte rien, et le prix d'exécution ne tient compte ni du volume ni de l'écart réel à l'ouverture.
Les positions ne sont pas rééquilibrées : quand les gagnantes ont absorbé le cash, un achat est
réduit ou ignoré plutôt que financé par de la marge.
"""

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
    StrategyConfig,
    load_strategy_config,
)
from bot.data.store import BarStore, StoreError
from bot.logs import configure_logging
from bot.strategy.base import LONG, Strategy
from bot.strategy.moving_average import MovingAverageCrossover

logger = structlog.get_logger()


class BacktestError(ValueError):
    """Données ou paramètres incompatibles avec le backtest."""


class BacktestReport(BaseModel):
    """Résultat complet d'un backtest in-sample."""

    model_config = ConfigDict(frozen=True)

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


def target_weights(signals: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Convertit les signaux en poids visés, en les décalant d'une séance.

    Le signal de t devient un ordre à l'ouverture de t+1. Une cellule n'est renseignée que si la
    position change : les positions évoluent ensuite librement, sans rééquilibrage ni frais.
    """
    executed = signals.shift(1).eq(LONG)
    changed = executed.ne(executed.shift(1, fill_value=False))
    return (executed.astype(float) * weight).where(changed)


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
    strategy: Strategy,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    weight: float,
    warm_up: int,
    config: BacktestConfig,
) -> BacktestRun:
    """Exécute une stratégie, évaluée dès la première séance où un ordre est possible."""
    _validate_prices(opens, closes, warm_up)
    weights = target_weights(strategy.generate(closes), weight)
    return _simulate(opens, closes, weights, config, warm_up)


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


def _variant_definition(config: StrategyConfig, strategy: Strategy) -> dict[str, Any]:
    return {
        "strategy": type(strategy).__name__,
        "params": config.strategy.model_dump(),
        "symbols": config.data.symbols,
        "start_date": config.data.start_date,
        "in_sample_end": config.backtest.in_sample_end,
        "fees_pct": config.backtest.fees_pct,
        "slippage_pct": config.backtest.slippage_pct,
    }


def run_in_sample(config: StrategyConfig, root: Path, now: datetime) -> BacktestReport:
    """Lance le backtest in-sample, écrit son rapport et l'inscrit au registre des variantes."""
    backtest = config.backtest
    symbols = config.data.symbols
    weight = 1 / len(symbols)
    if weight > config.risk.max_position_pct:
        raise BacktestError(f"Poids de {weight:.4f} supérieur à la limite par position")

    opens, closes = load_in_sample(BarStore(root / config.data.directory), config)
    strategy = MovingAverageCrossover(config.strategy)
    warm_up = config.strategy.slow_window
    run = run_strategy(strategy, opens, closes, weight, warm_up, backtest)
    benchmark_run = run_buy_and_hold(
        opens[backtest.benchmark], closes[backtest.benchmark], warm_up, backtest
    )

    metrics = compute_metrics(run.returns)
    benchmark = compute_metrics(benchmark_run.returns)
    definition = _variant_definition(config, strategy)
    report = BacktestReport(
        sample="in_sample",
        variant=variant_id(definition),
        start=pd.Timestamp(run.returns.index[0]).date(),
        end=pd.Timestamp(run.returns.index[-1]).date(),
        costs={"fees_pct": backtest.fees_pct, "slippage_pct": backtest.slippage_pct},
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

    report_path = root / backtest.report_file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    record_variant(
        root / backtest.registry_file,
        definition,
        {
            "annual_return": metrics.annual_return,
            "sharpe": metrics.sharpe,
            "max_drawdown": metrics.max_drawdown,
        },
        now,
    )
    return report


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
