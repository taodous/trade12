"""Page Streamlit du dashboard. Affichage seulement : les calculs sont dans `bot.dashboard.data`.

Lancement : `uv run python -m bot.dashboard`. La page lit le compte paper, le journal et les logs,
sans jamais envoyer d'ordre ni écrire. Elle peut être ouverte et fermée à tout moment.
"""

import math
from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bot.config import ConfigError, StrategyConfig, load_strategy_config
from bot.dashboard import data
from bot.execution.client import PaperModeError
from bot.status import StatusClient

POSITIVE = "#1a7f37"
NEGATIVE = "#cf222e"
NEUTRAL = "#57606a"
BENCHMARK = "#8c959f"
CHART_HEIGHT = 340


def _pct(value: float | None, decimals: int = 2) -> str:
    if value is None or not math.isfinite(value):
        return "n/d"
    return f"{value:.{decimals}%}"


def _usd(value: float) -> str:
    return f"{value:,.2f} $".replace(",", " ")


@st.cache_resource
def _config() -> StrategyConfig:
    return load_strategy_config()


@st.cache_resource
def _clients() -> tuple[StatusClient, data.DashboardDataClient]:
    return data.connect()


def _history(config: StrategyConfig) -> data.HistoryView:
    @st.cache_data(ttl=config.dashboard.history_cache_seconds, show_spinner=False)
    def cached(stamp: str) -> data.HistoryView:
        trading, market = _clients()
        return data.load_history(config, trading, market, datetime.now(UTC))

    # La clé change chaque jour : un nouveau jour de bourse recharge l'historique.
    return cached(datetime.now(UTC).date().isoformat())


def _header(live: data.LiveView, config: StrategyConfig) -> None:
    status = live.status
    columns = st.columns(6)
    columns[0].metric("Capital", _usd(status.equity), _pct(status.day_change))
    columns[1].metric("Exposition", _pct(status.exposure, 1))
    columns[2].metric("Cash", _usd(status.cash))
    columns[3].metric("Drawdown", _pct(status.drawdown))
    bot_state = "actif" if live.heartbeat.active else "arrêté"
    seen = live.heartbeat.last_seen
    columns[4].metric(
        "Bot", bot_state, f"dernier signe {seen:%H:%M:%S} UTC" if seen else "jamais vu", "off"
    )
    columns[5].metric("Marché", "ouvert" if live.clock.is_open else "fermé")

    risk = status.risk
    if risk.halt_reason is not None:
        halted = f"{risk.halted_at:%Y-%m-%d %H:%M} UTC" if risk.halted_at else "?"
        st.error(
            f"**Kill switch déclenché** : {risk.halt_reason.value}, le {halted}. Le bot ferme les "
            "positions et refuse tout ordre. Reprise sur décision humaine uniquement : "
            "`uv run python -m bot.main --reset-kill-switch`."
        )
    limits = config.risk
    day_loss = max(0.0, -status.day_change)
    gauges = st.columns(2)
    gauges[0].progress(
        min(1.0, day_loss / limits.max_daily_loss_pct),
        text=f"Perte du jour {_pct(day_loss)} sur {_pct(limits.max_daily_loss_pct)} (kill switch)",
    )
    gauges[1].progress(
        min(1.0, status.drawdown / limits.max_drawdown_pct),
        text=f"Drawdown {_pct(status.drawdown)} sur {_pct(limits.max_drawdown_pct)} (arrêt)",
    )
    for note in status.notes:
        st.warning(note)


def _intraday(live: data.LiveView) -> None:
    series = live.intraday
    if series.empty:
        st.info("Pas encore de données intraday pour cette séance.")
        return
    base = live.status.last_equity
    color = POSITIVE if series.iloc[-1] >= base else NEGATIVE
    figure = go.Figure(
        go.Scatter(x=series.index, y=series, mode="lines", line={"color": color}, name="capital")
    )
    figure.add_hline(y=base, line={"dash": "dot", "color": NEUTRAL}, annotation_text="veille")
    figure.update_layout(
        height=CHART_HEIGHT, margin={"t": 10, "b": 10}, yaxis_title="$", xaxis_title="New York"
    )
    st.plotly_chart(figure, width="stretch")


def _positions(live: data.LiveView) -> None:
    if not live.positions:
        st.info("Aucune position.")
        return
    targets = {line.symbol: line for line in live.status.positions}
    equity = live.status.equity
    rows = []
    for item in live.positions:
        line = targets.get(item.symbol)
        rows.append(
            {
                "symbole": item.symbol,
                "parts": item.quantity,
                "prix moyen": item.avg_entry_price,
                "prix": item.current_price,
                "valeur": item.market_value,
                "P&L latent $": item.unrealized_pl,
                "P&L latent": item.unrealized_plpc,
                "jour": item.change_today,
                "poids": item.market_value / equity,
                "poids visé": line.target_weight if line else None,
                "univers": "oui" if line is None or line.in_universe else "HORS UNIVERS",
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "parts": "{:g}",
                "prix moyen": "{:.2f}",
                "prix": "{:.2f}",
                "valeur": "{:,.2f}",
                "P&L latent $": "{:+,.2f}",
                "P&L latent": "{:+.2%}",
                "jour": "{:+.2%}",
                "poids": "{:.1%}",
                "poids visé": "{:.1%}",
            },
            na_rep="n/d",
        ),
        hide_index=True,
        width="stretch",
    )
    figure = go.Figure(
        [
            go.Bar(name="poids réel", x=table["symbole"], y=table["poids"], marker_color=NEUTRAL),
            go.Bar(
                name="poids visé", x=table["symbole"], y=table["poids visé"], marker_color=BENCHMARK
            ),
        ]
    )
    figure.update_layout(
        barmode="group", height=CHART_HEIGHT, margin={"t": 10, "b": 10}, yaxis_tickformat=".0%"
    )
    st.plotly_chart(figure, width="stretch")


def _decisions(config: StrategyConfig) -> None:
    journal = data.read_journal(config)
    if not journal.decisions:
        st.info("Journal vide : le bot n'a encore pris aucune décision.")
        return
    slippage, fills = data.mean_slippage(journal.decisions)
    st.caption(
        f"Slippage moyen observé : {_pct(slippage, 3)} sur {fills} exécutions, contre "
        f"{_pct(config.backtest.slippage_pct, 2)} supposé par le backtest. Alpaca simule les "
        "exécutions paper : le slippage réel serait plus élevé."
    )
    table = data.decision_table(journal.decisions, config.dashboard.recent_decisions)
    st.dataframe(
        table.style.format(
            {
                "heure (UTC)": lambda stamp: f"{stamp:%Y-%m-%d %H:%M:%S}",
                "poids visé": "{:.1%}",
                "prix exécuté": "{:.2f}",
                "slippage": "{:+.3%}",
            },
            na_rep="",
        ),
        hide_index=True,
        width="stretch",
    )


def _logs(live: data.LiveView) -> None:
    log = live.status.log
    if log.path is None:
        st.info("Aucun fichier de log.")
        return
    st.write(f"`{log.path.name}` : {log.errors} erreurs, {log.warnings} avertissements.")
    if log.unreadable:
        st.warning(f"{log.unreadable} lignes illisibles.")
    if log.last_error:
        st.error(f"Dernière erreur : {log.last_error}")


def _live(config: StrategyConfig) -> None:
    try:
        trading, _ = _clients()
        live = data.load_live(config, trading, datetime.now(UTC))
    except data.DASHBOARD_ERRORS as exc:
        st.error(f"Lecture du compte impossible ({type(exc).__name__}) : nouvel essai bientôt.")
        return
    _header(live, config)
    if live.status.missed_sessions:
        days = ", ".join(day.isoformat() for day in live.status.missed_sessions)
        st.warning(f"Séances sans bot actif : {days}.")
    today, positions, decisions, logs = st.tabs(["Aujourd'hui", "Positions", "Décisions", "Logs"])
    with today:
        _intraday(live)
    with positions:
        _positions(live)
    with decisions:
        _decisions(config)
    with logs:
        _logs(live)
    st.caption(
        f"Mis à jour à {datetime.now(UTC):%H:%M:%S} UTC, toutes les "
        f"{config.dashboard.refresh_seconds} s."
    )


def _equity_chart(history: data.HistoryView) -> None:
    figure = go.Figure(
        [
            go.Scatter(
                x=history.equity.index,
                y=history.equity,
                name="bot V1",
                mode="lines+markers",
                line={"color": POSITIVE},
            ),
            go.Scatter(
                x=history.benchmark.index,
                y=history.benchmark,
                name="SPY, même capital",
                mode="lines",
                line={"color": BENCHMARK, "dash": "dash"},
            ),
        ]
    )
    figure.update_layout(height=CHART_HEIGHT, margin={"t": 10, "b": 10}, yaxis_title="$")
    st.plotly_chart(figure, width="stretch")


def _drawdown_chart(history: data.HistoryView, config: StrategyConfig) -> None:
    figure = go.Figure(
        go.Scatter(
            x=history.drawdown.index,
            y=-history.drawdown,
            fill="tozeroy",
            name="drawdown",
            line={"color": NEGATIVE},
        )
    )
    figure.add_hline(
        y=-config.risk.max_drawdown_pct,
        line={"dash": "dot", "color": NEGATIVE},
        annotation_text="arrêt du bot",
    )
    figure.update_layout(height=CHART_HEIGHT, margin={"t": 10, "b": 10}, yaxis_tickformat=".1%")
    st.plotly_chart(figure, width="stretch")


def _metrics_table(history: data.HistoryView, config: StrategyConfig) -> None:
    bot, spy, targets = history.metrics, history.benchmark_metrics, config.targets

    def row(name: str, attribute: str, target: str) -> dict[str, str]:
        return {
            "métrique": name,
            "bot": _pct(getattr(bot, attribute)) if bot else "n/d",
            "SPY": _pct(getattr(spy, attribute)) if spy else "n/d",
            "cible": target,
        }

    rows = [
        {
            "métrique": "Rendement depuis le début",
            "bot": _pct(history.total_return),
            "SPY": _pct(history.benchmark_return),
            "cible": "",
        },
        row("Rendement annualisé", "annual_return", f"≥ {_pct(targets.min_annual_return, 0)}"),
        row("Volatilité annualisée", "annual_volatility", "< SPY"),
        row("Max drawdown", "max_drawdown", f"< {_pct(targets.max_drawdown_alert, 0)}"),
        row("Mois positifs", "positive_months", f"≥ {_pct(targets.min_positive_months, 0)}"),
        {
            "métrique": "Sharpe",
            "bot": f"{bot.sharpe:.2f}" if bot and math.isfinite(bot.sharpe) else "n/d",
            "SPY": f"{spy.sharpe:.2f}" if spy and math.isfinite(spy.sharpe) else "n/d",
            "cible": f"≥ {targets.min_sharpe:g} et > SPY",
        },
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _performance(config: StrategyConfig) -> None:
    st.subheader("Performance depuis le début du paper")
    try:
        history = _history(config)
    except data.DASHBOARD_ERRORS as exc:
        st.error(f"Historique indisponible ({type(exc).__name__}) : nouvel essai bientôt.")
        return
    if history.start is None or history.equity.empty:
        st.info("Pas encore d'historique : le bot n'a tourné aucune séance.")
        return
    sessions = len(history.equity) - 1
    st.caption(
        f"Depuis le {history.start:%d/%m/%Y}, {sessions} séance(s), "
        f"{len(history.bot_sessions)} avec le bot actif. Capital de clôture, et celui du jour "
        "en direct."
    )
    if sessions < config.dashboard.min_sessions_for_metrics:
        st.warning(
            f"Échantillon de {sessions} séance(s) : les chiffres annualisés ne sont pas "
            f"significatifs avant environ {config.dashboard.min_sessions_for_metrics} séances "
            "(3 mois)."
        )
    left, right = st.columns([3, 2])
    with left:
        _equity_chart(history)
    with right:
        _drawdown_chart(history, config)
    _metrics_table(history, config)
    st.caption(
        "Comparaison au backtest de V1 sur la même période : à venir (partie B de l'étape 9)."
    )


def render() -> None:
    """Construit la page."""
    st.set_page_config(page_title="Bot paper V1", page_icon=":material/monitoring:", layout="wide")
    try:
        config = _config()
    except (ConfigError, PaperModeError, OSError) as exc:
        st.error(f"Configuration illisible : {exc}")
        return
    st.title("Bot de trading, compte paper")
    st.caption(
        f"Stratégie V1, lecture seule, compte PAPER Alpaca. Benchmark {config.backtest.benchmark}."
    )

    live = st.fragment(run_every=config.dashboard.refresh_seconds)(_live)
    live(config)
    st.divider()
    history = st.fragment(run_every=config.dashboard.history_cache_seconds)(_performance)
    history(config)


render()
