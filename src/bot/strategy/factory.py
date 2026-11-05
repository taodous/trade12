"""Construction de la stratégie décrite par la configuration."""

from collections.abc import Sequence

from bot.config import MovingAverageSettings, RebalanceConfig, RiskTargetedSettings
from bot.strategy.base import WeightStrategy
from bot.strategy.equal_weight import EqualWeight
from bot.strategy.moving_average import MovingAverageCrossover
from bot.strategy.risk_targeted import RiskTargeted

# Noms inscrits au registre des variantes. Celui de la stratégie de référence ne doit pas changer,
# sinon ses variantes déjà testées seraient comptées une seconde fois.
STRATEGY_LABELS = {"moving_average": "MovingAverageCrossover", "risk_targeted": "RiskTargeted"}


def build_strategy(
    settings: MovingAverageSettings | RiskTargetedSettings, symbols: Sequence[str]
) -> WeightStrategy:
    """Retourne la stratégie configurée pour cet univers."""
    if isinstance(settings, MovingAverageSettings):
        return EqualWeight(
            MovingAverageCrossover(settings.params),
            weight=1 / len(symbols),
            warm_up=settings.params.slow_window,
        )
    return RiskTargeted(settings.params)


def rebalance_policy(
    settings: MovingAverageSettings | RiskTargetedSettings,
) -> RebalanceConfig | None:
    """Retourne le rééquilibrage de la stratégie, ou None si les positions évoluent librement."""
    return settings.rebalance if isinstance(settings, RiskTargetedSettings) else None
