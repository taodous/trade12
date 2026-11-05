"""Test de sabotage : chaque contrôle de sûreté cassé doit faire échouer au moins un test.

Un sabotage remplace un fragment de code par une version fautive, puis lance les tests. S'ils
passent encore, le contrôle n'est vérifié par aucun test : le sabotage a survécu.

Le travail se fait sur une copie temporaire du projet, importée à la place du code réel. Un arrêt
brutal en plein sabotage ne peut donc pas laisser de code cassé dans le dépôt. Chaque lancement a
une limite de temps : un sabotage qui fait tourner un test sans fin est arrêté, et compté comme
détecté puisque ce test n'aurait jamais abouti.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict

from bot.config import PROJECT_ROOT, ConfigError, SabotageConfig, load_strategy_config
from bot.logs import configure_logging

logger = structlog.get_logger()

COPIED_DIRECTORIES = ("src", "tests", "audit", "config")
COPIED_FILES = ("pyproject.toml",)
HASHED_DIRECTORIES = ("src", "tests")


class SabotageError(RuntimeError):
    """Sabotage inapplicable ou environnement de test incorrect."""


@dataclass(frozen=True)
class Sabotage:
    """Remplacement d'un fragment de code, qui doit apparaître exactement une fois."""

    name: str
    path: str
    original: str
    replacement: str
    description: str


CHECKS = "src/bot/risk/checks.py"
MANAGER = "src/bot/risk/manager.py"
MAIN = "src/bot/main.py"
BROKER = "src/bot/execution/broker.py"
PLANNER = "src/bot/execution/planner.py"
RISK_TARGETED = "src/bot/strategy/risk_targeted.py"

SABOTAGES: tuple[Sabotage, ...] = (
    Sabotage(
        "daily_loss_disabled",
        CHECKS,
        "if portfolio.equity <= portfolio.day_start_equity * (1 - limits.max_daily_loss_pct):",
        "if False:",
        "La perte journalière ne déclenche plus le kill switch.",
    ),
    Sabotage(
        "daily_loss_boundary",
        CHECKS,
        "if portfolio.equity <= portfolio.day_start_equity * (1 - limits.max_daily_loss_pct):",
        "if portfolio.equity < portfolio.day_start_equity * (1 - limits.max_daily_loss_pct):",
        "Une perte exactement égale à la limite ne déclenche plus le kill switch.",
    ),
    Sabotage(
        "drawdown_disabled",
        CHECKS,
        "if portfolio.equity <= peak_equity * (1 - limits.max_drawdown_pct):",
        "if False:",
        "Le drawdown maximal ne déclenche plus le kill switch.",
    ),
    Sabotage(
        "peak_not_kept",
        MANAGER,
        "peak_equity = max(self._state.peak_equity or portfolio.equity, portfolio.equity)",
        "peak_equity = portfolio.equity",
        "Le plus haut du capital n'est plus conservé, le drawdown ne peut plus se déclencher.",
    ),
    Sabotage(
        "halt_not_recorded",
        MANAGER,
        "if reason is not None and not self.is_halted:",
        "if False:",
        "Une limite franchie n'arme plus le kill switch.",
    ),
    Sabotage(
        "halt_ignored_in_review",
        MANAGER,
        "        if self._state.halt_reason is not None:\n",
        "        if False:\n",
        "Le kill switch armé ne bloque plus les ordres.",
    ),
    Sabotage(
        "invalid_portfolio_accepted",
        MANAGER,
        "        if not is_valid_portfolio(portfolio):\n            return RiskDecision(",
        "        if False:\n            return RiskDecision(",
        "Un état de compte inexploitable n'empêche plus les ordres.",
    ),
    Sabotage(
        "order_limit_off_by_one",
        MANAGER,
        "if sent >= self._limits.max_orders_per_day:",
        "if sent > self._limits.max_orders_per_day:",
        "Un ordre de plus que la limite journalière est accepté.",
    ),
    Sabotage(
        "invalid_quantity_accepted",
        CHECKS,
        "if not (math.isfinite(order.quantity) and order.quantity > 0):",
        "if False:",
        "Une quantité nulle, négative ou NaN est acceptée.",
    ),
    Sabotage(
        "missing_price_accepted",
        CHECKS,
        '    if snapshot is None:\n        return _reject(RejectReason.MISSING_DATA, "aucun prix")',
        "    if snapshot is None:\n        return APPROVED",
        "Un ordre sans prix est accepté.",
    ),
    Sabotage(
        "naive_timestamp_accepted",
        CHECKS,
        "if snapshot.as_of.tzinfo is None:",
        "if False:",
        "Un prix sans fuseau horaire est accepté.",
    ),
    Sabotage(
        "non_positive_price_accepted",
        CHECKS,
        "if not (math.isfinite(price) and price > 0):",
        "if not math.isfinite(price):",
        "Un prix nul ou négatif est accepté.",
    ),
    Sabotage(
        "future_price_accepted",
        CHECKS,
        "if age < -limits.max_clock_skew_seconds:",
        "if False:",
        "Un prix daté du futur est accepté.",
    ),
    Sabotage(
        "stale_price_accepted",
        CHECKS,
        "if age > limits.max_data_age_seconds:",
        "if False:",
        "Un prix périmé est accepté.",
    ),
    Sabotage(
        "downward_gap_accepted",
        CHECKS,
        "if abs(change) > limits.max_price_change_pct:",
        "if change > limits.max_price_change_pct:",
        "Une chute aberrante du prix, comme un gap à l'ouverture, est acceptée.",
    ),
    Sabotage(
        "stale_signal_accepted",
        CHECKS,
        "if order.signal_session != context.last_completed_session:",
        "if order.signal_session > context.last_completed_session:",
        "Un signal ancien est accepté.",
    ),
    Sabotage(
        "short_accepted",
        CHECKS,
        "if projected < 0 and (current >= 0 or increases_risk) and not limits.allow_short:",
        "if False:",
        "La vente à découvert est acceptée sans activation.",
    ),
    Sabotage(
        "position_limit_disabled",
        CHECKS,
        "if abs(projected) > position_limit:",
        "if False:",
        "La taille maximale par position n'est plus appliquée.",
    ),
    Sabotage(
        "pending_orders_ignored",
        CHECKS,
        "projected = (position.quantity + signed_quantity) * price + position.pending_notional",
        "projected = (position.quantity + signed_quantity) * price",
        "Les ordres ouverts ne comptent plus dans la position visée.",
    ),
    Sabotage(
        "exposure_limit_disabled",
        CHECKS,
        "if others + abs(projected) > exposure_limit:",
        "if False:",
        "L'exposition totale maximale n'est plus appliquée.",
    ),
    Sabotage(
        "short_positions_not_closed",
        CHECKS,
        "if position.quantity != 0",
        "if position.quantity > 0",
        "Le kill switch ne ferme plus les positions à découvert.",
    ),
    Sabotage(
        "closing_side_wrong",
        CHECKS,
        "side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,",
        "side=OrderSide.SELL,",
        "Le kill switch vend une position à découvert au lieu de la racheter.",
    ),
    Sabotage(
        "risk_bypassed",
        MAIN,
        "        if not decision.approved:\n            logger.warning(",
        "        if False:\n            logger.warning(",
        "Un ordre refusé par le module risque est envoyé quand même.",
    ),
    Sabotage(
        "kill_switch_does_not_liquidate",
        MAIN,
        "            self._liquidate(now)\n",
        "",
        "Le kill switch armé ne ferme plus les positions.",
    ),
    Sabotage(
        "cycle_repeated",
        MAIN,
        "if not self._dry_run and self._journal.is_cycle_completed(session):",
        "if False:",
        "Une séance déjà traitée est traitée à nouveau, au risque de doubler les ordres.",
    ),
    Sabotage(
        "market_closed_ignored",
        MAIN,
        "if not clock.is_open and not self._dry_run:",
        "if False:",
        "Le bot passe des ordres marché fermé.",
    ),
    Sabotage(
        "account_status_ignored",
        MAIN,
        "if not account.can_trade:",
        "if False:",
        "Le bot passe des ordres sur un compte bloqué.",
    ),
    Sabotage(
        "missing_close_used",
        MAIN,
        "complete = closes.columns[closes.tail(warm_up).notna().all()]",
        "complete = closes.columns",
        "Une clôture manquante produit un signal, donc une vente à tort.",
    ),
    Sabotage(
        "pending_order_not_counted",
        MAIN,
        "return _with_pending(portfolio, order, price)",
        "return portfolio",
        "Les ordres d'un cycle ne comptent plus dans l'exposition des suivants.",
    ),
    Sabotage(
        "api_failures_never_stop",
        MAIN,
        "if once or failures >= self._config.live.max_consecutive_errors:",
        "if once:",
        "Le bot ne s'arrête plus après des pannes répétées de l'API.",
    ),
    Sabotage(
        "missing_signal_sells",
        PLANNER,
        "    if target is None:\n        return PlanItem(symbol, target, PlanNote.MISSING_DATA)\n",
        "",
        "Un signal manquant est traité comme une sortie de position.",
    ),
    Sabotage(
        "rebalance_band_ignored",
        PLANNER,
        "if abs(change) * price < rebalance.band_pct * portfolio.equity:",
        "if False:",
        "Le moindre écart de poids produit un ordre, au lieu de rester dans la bande.",
    ),
    Sabotage(
        "rebalance_every_session",
        MAIN,
        "and self._journal.has_completed_cycle_in_month(session)",
        "and False",
        "Une stratégie mensuelle est rééquilibrée à chaque séance, contrairement au backtest.",
    ),
    Sabotage(
        "required_symbol_ignored",
        MAIN,
        "if len(complete) and required <= {str(symbol) for symbol in complete}:",
        "if len(complete):",
        "Les poids sont calculés sans le symbole du filtre de régime.",
    ),
    Sabotage(
        "max_weight_ignored",
        RISK_TARGETED,
        "return np.asarray(np.minimum(base * scale, params.max_weight))",
        "return np.asarray(base * scale)",
        "Le plafond par position de la stratégie n'est plus appliqué.",
    ),
    Sabotage(
        "max_exposure_ignored",
        RISK_TARGETED,
        "scale = min(params.target_volatility / portfolio_volatility, params.max_exposure)",
        "scale = params.target_volatility / portfolio_volatility",
        "La volatilité cible peut porter l'exposition au-delà du capital.",
    ),
    Sabotage(
        "trend_filter_ignored",
        RISK_TARGETED,
        "active = trend[row] & np.isfinite(window).all(axis=0)",
        "active = np.isfinite(window).all(axis=0)",
        "Un actif en tendance baissière reste détenu.",
    ),
    Sabotage(
        "regime_filter_ignored",
        RISK_TARGETED,
        "result.loc[risk_off, risky] *= params.regime_scale",
        "result.loc[risk_off, risky] *= 1.0",
        "Le filtre de régime ne réduit plus l'exposition.",
    ),
    Sabotage(
        "volatility_look_ahead",
        RISK_TARGETED,
        "window = returns[row - window_length + 1 : row + 1]",
        "window = returns[row - window_length + 2 : row + 2]",
        "La volatilité utilise le rendement de la séance suivante.",
    ),
    Sabotage(
        "duplicate_reported_as_rejection",
        BROKER,
        "            if existing is not None:\n                return existing\n",
        "",
        "Un identifiant déjà utilisé n'est plus reconnu comme l'ordre existant.",
    ),
    Sabotage(
        "order_id_not_stable",
        BROKER,
        'return f"{prefix}-{session:%Y%m%d}-{symbol}-{side.value}"',
        'return f"{prefix}-{session:%Y%m%d}-{symbol}-{side.value}-'
        "{__import__('time').perf_counter_ns()}\"",
        "L'identifiant d'ordre change à chaque envoi, ce qui permet les doublons.",
    ),
)


class Outcome(StrEnum):
    """Résultat d'un sabotage."""

    DETECTED = "detected"
    TIMEOUT = "timeout"
    SURVIVED = "survived"


class SabotageResult(BaseModel):
    """Résultat d'un sabotage."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    name: str
    path: str
    description: str
    outcome: Outcome
    seconds: float


class SabotageReport(BaseModel):
    """Résultat complet, lié à l'état du code par `source_hash`."""

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="strings")

    ran_at: datetime
    source_hash: str
    baseline_seconds: float
    timeout_seconds: float
    results: list[SabotageResult]

    @property
    def survivors(self) -> list[str]:
        """Retourne les sabotages qu'aucun test n'a détectés."""
        return [item.name for item in self.results if item.outcome is Outcome.SURVIVED]


def apply_sabotage(text: str, sabotage: Sabotage) -> str:
    """Retourne le code saboté, ou lève `SabotageError` si le fragment n'est pas unique."""
    count = text.count(sabotage.original)
    if count != 1:
        raise SabotageError(f"{sabotage.name} : fragment trouvé {count} fois dans {sabotage.path}")
    return text.replace(sabotage.original, sabotage.replacement)


def source_hash(root: Path) -> str:
    """Retourne une empreinte du code et des tests, pour repérer un résultat périmé."""
    digest = hashlib.sha256()
    for directory in HASHED_DIRECTORIES:
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            # Les fins de ligne dépendent de la configuration git, pas du code.
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _copy_project(root: Path, destination: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for directory in COPIED_DIRECTORIES:
        shutil.copytree(root / directory, destination / directory, ignore=ignore)
    for name in COPIED_FILES:
        shutil.copy2(root / name, destination / name)


def _environment(workdir: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(workdir / "src"), str(workdir)])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _check_isolation(workdir: Path) -> None:
    """Vérifie que la copie, et non le code réel, est importée."""
    script = "import audit, bot; print(bot.__file__); print(audit.__file__)"
    command = [sys.executable, "-c", script]
    completed = subprocess.run(
        command, cwd=workdir, env=_environment(workdir), capture_output=True, text=True, check=False
    )
    paths = [Path(line).resolve() for line in completed.stdout.split()]
    if completed.returncode != 0 or len(paths) != 2:
        raise SabotageError(f"Import impossible dans la copie : {completed.stderr.strip()}")
    if not all(path.is_relative_to(workdir.resolve()) for path in paths):
        raise SabotageError(f"La copie n'est pas importée : {paths}")


def _run_tests(workdir: Path, settings: SabotageConfig, timeout: float | None) -> tuple[int, float]:
    """Lance les tests et retourne le code de retour, ou -1 si la limite de temps est atteinte.

    La sortie part vers le néant : un test sans fin qui journalise en boucle ne doit pas remplir
    la mémoire.
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-x",
        "-q",
        "-s",
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        *settings.test_paths,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=_environment(workdir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return -1, time.monotonic() - started
    return completed.returncode, time.monotonic() - started


def run_sabotage(
    root: Path, settings: SabotageConfig, now: datetime, sabotages: tuple[Sabotage, ...] = SABOTAGES
) -> SabotageReport:
    """Applique chaque sabotage sur une copie du projet et lance les tests."""
    with tempfile.TemporaryDirectory(prefix="sabotage-") as directory:
        workdir = Path(directory)
        _copy_project(root, workdir)
        # Empreinte de la copie, donc du code réellement testé, même s'il change pendant le test.
        tested_hash = source_hash(workdir)
        _check_isolation(workdir)
        originals = {item.path: (workdir / item.path).read_text("utf-8") for item in sabotages}
        sabotaged = {item.name: apply_sabotage(originals[item.path], item) for item in sabotages}

        code, baseline = _run_tests(workdir, settings, settings.baseline_timeout_seconds)
        if code != 0:
            raise SabotageError(f"Les tests échouent sans sabotage (code {code})")
        timeout = max(settings.min_timeout_seconds, settings.timeout_multiplier * baseline)
        logger.info("sabotage_baseline", seconds=round(baseline, 1), timeout=round(timeout, 1))

        results = []
        for item in sabotages:
            target = workdir / item.path
            target.write_text(sabotaged[item.name], encoding="utf-8")
            try:
                code, seconds = _run_tests(workdir, settings, timeout)
            finally:
                target.write_text(originals[item.path], encoding="utf-8")
            if code == -1:
                outcome = Outcome.TIMEOUT
            else:
                outcome = Outcome.DETECTED if code != 0 else Outcome.SURVIVED
            logger.info("sabotage_done", name=item.name, outcome=outcome.value)
            results.append(
                SabotageResult(
                    name=item.name,
                    path=item.path,
                    description=item.description,
                    outcome=outcome,
                    seconds=seconds,
                )
            )

    return SabotageReport(
        ran_at=now,
        source_hash=tested_hash,
        baseline_seconds=baseline,
        timeout_seconds=timeout,
        results=results,
    )


def load_sabotage_report(path: Path) -> SabotageReport | None:
    """Relit le dernier résultat, ou None s'il n'existe pas."""
    if not path.exists():
        return None
    return SabotageReport.model_validate_json(path.read_text(encoding="utf-8"))


def main() -> int:
    """Lance le test de sabotage, écrit son résultat et retourne le code de sortie."""
    configure_logging()
    try:
        config = load_strategy_config()
        report = run_sabotage(PROJECT_ROOT, config.audit.sabotage, datetime.now(UTC))
    except (ConfigError, SabotageError) as exc:
        logger.error("sabotage_failed", reason=str(exc))
        return 1
    path = PROJECT_ROOT / config.audit.sabotage.report_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("sabotage_report", survivors=report.survivors, total=len(report.results))
    return 0 if not report.survivors else 1


if __name__ == "__main__":
    sys.exit(main())
