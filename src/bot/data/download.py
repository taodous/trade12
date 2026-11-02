"""Télécharge, contrôle et stocke l'historique de l'univers configuré."""

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog
from alpaca.common.exceptions import APIError
from requests.exceptions import RequestException

from bot.config import (
    PROJECT_ROOT,
    ConfigError,
    StrategyConfig,
    load_settings,
    load_strategy_config,
)
from bot.data.market import (
    BarsClient,
    CalendarClient,
    DataFetchError,
    create_data_client,
    fetch_bars,
    fetch_sessions,
    last_completed_session,
)
from bot.data.quality import clean_bars
from bot.data.store import BarStore, StoreError
from bot.execution.client import PaperModeError, create_trading_client
from bot.logs import configure_logging

logger = structlog.get_logger()


def download_universe(
    config: StrategyConfig,
    bars_client: BarsClient,
    calendar_client: CalendarClient,
    store: BarStore,
    *,
    now: datetime,
    overwrite: bool = False,
) -> None:
    """Télécharge chaque symbole, le contrôle et le stocke.

    Lève `StoreError` avant tout téléchargement si un jeu existant serait remplacé sans demande.
    """
    data = config.data
    if not overwrite:
        existing = [symbol for symbol in data.symbols if store.exists(symbol)]
        if existing:
            raise StoreError(f"Des barres existent déjà pour : {', '.join(existing)}")

    end_date = data.end_date or now.date()
    sessions = fetch_sessions(calendar_client, data.start_date, end_date, config.api_retry)
    last = last_completed_session(sessions, now, timedelta(minutes=data.min_bar_age_minutes))
    expected = [session.date for session in sessions if session.date <= last.date]

    for symbol in data.symbols:
        raw = fetch_bars(bars_client, symbol, data.start_date, last.close, data, config.api_retry)
        cleaned = clean_bars(raw, expected, config.risk.max_price_change_pct)
        metadata = store.save(
            symbol,
            cleaned.bars,
            feed=data.feed,
            adjustment=data.adjustment,
            fetched_at=now,
            issues=cleaned.issues,
            overwrite=overwrite,
        )
        logger.info(
            "bars_stored",
            symbol=symbol,
            rows=metadata.rows,
            start=metadata.start.isoformat(),
            end=metadata.end.isoformat(),
            issues=len(metadata.issues),
        )
        for issue in metadata.issues:
            logger.warning(
                "data_issue",
                symbol=symbol,
                session=issue.session.isoformat(),
                kind=issue.kind.value,
                removed=issue.removed,
                detail=issue.detail,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Lance le téléchargement et retourne le code de sortie du processus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="remplace les jeux existants, ce qui change les données des backtests futurs",
    )
    args = parser.parse_args(argv)

    configure_logging()
    try:
        config = load_strategy_config()
        settings = load_settings()
        download_universe(
            config,
            create_data_client(settings),
            create_trading_client(settings),
            BarStore(PROJECT_ROOT / Path(config.data.directory)),
            now=datetime.now(UTC),
            overwrite=args.overwrite,
        )
    except (ConfigError, PaperModeError, DataFetchError, StoreError) as exc:
        logger.error("download_failed", reason=str(exc))
        return 1
    except APIError as exc:
        logger.error("download_failed", reason="api_error", status_code=exc.status_code)
        return 1
    except RequestException as exc:
        logger.error("download_failed", reason="network_error", error=type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
