"""Seul point de contact avec l'API d'ordres d'Alpaca."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from http import HTTPStatus
from typing import Protocol
from zoneinfo import ZoneInfo

from alpaca.common.exceptions import APIError
from alpaca.common.types import RawData
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderStatus, PositionSide, QueryOrderStatus, TimeInForce
from alpaca.trading.models import Clock, Order, PortfolioHistory
from alpaca.trading.models import Position as AlpacaPosition
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    MarketOrderRequest,
    OrderRequest,
)

from bot.config import RetryConfig
from bot.data.market import EXCHANGE_TIMEZONE
from bot.execution.client import AccountClient, AccountSnapshot, read_account
from bot.retry import with_backoff
from bot.risk.models import OrderSide, PortfolioState, Position

FINAL_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.EXPIRED,
        OrderStatus.REJECTED,
        OrderStatus.DONE_FOR_DAY,
        OrderStatus.REPLACED,
        OrderStatus.SUSPENDED,
    }
)
REJECTED_BY_BROKER = "rejected_by_broker"


class BrokerError(RuntimeError):
    """Réponse de l'API d'ordres inexploitable."""


class BrokerClient(AccountClient, Protocol):
    """Partie de l'API de trading utilisée par le bot."""

    def get_clock(self) -> Clock | RawData:
        """Retourne l'état du marché."""
        ...

    def get_all_positions(self) -> list[AlpacaPosition] | RawData:
        """Retourne les positions."""
        ...

    def get_orders(self, filter: GetOrdersRequest | None = None) -> list[Order] | RawData:
        """Retourne les ordres demandés."""
        ...

    def submit_order(self, order_data: OrderRequest) -> Order | RawData:
        """Envoie un ordre."""
        ...

    def get_order_by_client_id(self, client_id: str) -> Order | RawData:
        """Retourne un ordre d'après son identifiant client."""
        ...

    def cancel_orders(self) -> object:
        """Annule tous les ordres ouverts."""
        ...

    def get_portfolio_history(
        self, history_filter: GetPortfolioHistoryRequest | None = None
    ) -> PortfolioHistory | RawData:
        """Retourne l'historique du capital du compte."""
        ...


@dataclass(frozen=True)
class MarketClock:
    """État du marché selon le broker."""

    is_open: bool
    timestamp: datetime
    next_open: datetime
    next_close: datetime

    @property
    def is_session_over(self) -> bool:
        """Indique si la séance du jour est terminée, ou s'il n'y en a pas aujourd'hui.

        Avant l'ouverture, la prochaine ouverture tombe le jour même : la séance reste à venir.
        Les jours fériés et les changements d'heure viennent ainsi de l'horloge d'Alpaca.
        """
        exchange = ZoneInfo(EXCHANGE_TIMEZONE)
        today = self.timestamp.astimezone(exchange).date()
        return not self.is_open and self.next_open.astimezone(exchange).date() > today


@dataclass(frozen=True)
class OpenOrder:
    """Ordre encore ouvert. `signed_quantity` est le reste à exécuter, négatif à la vente."""

    symbol: str
    signed_quantity: float


@dataclass(frozen=True)
class OrderReport:
    """État d'un ordre chez le broker."""

    client_order_id: str
    broker_order_id: str | None
    status: str
    is_final: bool
    filled_quantity: float = 0.0
    filled_price: float | None = None
    detail: str = ""


@dataclass(frozen=True)
class PositionDetail:
    """Position du compte, avec son prix de revient et son résultat latent, pour le suivi."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    change_today: float


@dataclass(frozen=True)
class EquitySample:
    """Capital du compte à la fin d'une fenêtre de l'historique Alpaca."""

    timestamp: datetime
    equity: float


def make_client_order_id(prefix: str, session: date, symbol: str, side: OrderSide) -> str:
    """Retourne un identifiant stable pour un ordre d'une séance.

    Alpaca refuse deux ordres portant le même identifiant : un redémarrage ou une réponse perdue
    ne peut donc pas envoyer deux fois le même ordre.
    """
    return f"{prefix}-{session:%Y%m%d}-{symbol}-{side.value}"


def _to_float(value: str | float | None) -> float:
    return float(value) if value is not None else 0.0


def _or_nan(value: str | float | None) -> float:
    """Une valeur absente reste inconnue : NaN, et non 0, pour ne pas fausser un poids."""
    return float(value) if value is not None else math.nan


def _report(order: Order) -> OrderReport:
    price = order.filled_avg_price
    return OrderReport(
        client_order_id=order.client_order_id,
        broker_order_id=str(order.id),
        status=order.status.value,
        is_final=order.status in FINAL_STATUSES,
        filled_quantity=_to_float(order.filled_qty),
        filled_price=float(price) if price is not None else None,
    )


class Broker:
    """Accès à Alpaca, avec backoff sur erreur 429 et réponses typées."""

    def __init__(self, client: BrokerClient, retry: RetryConfig) -> None:
        """Associe le broker à un client de trading paper."""
        self._client = client
        self._retry = retry

    def clock(self) -> MarketClock:
        """Retourne l'état du marché."""
        clock = with_backoff(self._client.get_clock, self._retry)
        if not isinstance(clock, Clock):
            raise BrokerError("Réponse brute reçue au lieu de l'horloge")
        return MarketClock(clock.is_open, clock.timestamp, clock.next_open, clock.next_close)

    def account(self) -> AccountSnapshot:
        """Retourne l'état du compte."""
        return with_backoff(lambda: read_account(self._client), self._retry)

    def account_summary(self) -> PortfolioState:
        """Retourne le capital sans les positions, ce qui suffit au contrôle du kill switch."""
        account = self.account()
        return PortfolioState(float(account.equity), float(account.last_equity))

    def positions(self) -> dict[str, Position]:
        """Retourne les positions, avec une quantité négative pour une vente à découvert."""
        positions = with_backoff(self._client.get_all_positions, self._retry)
        if isinstance(positions, dict):
            raise BrokerError("Réponse brute reçue au lieu des positions")
        result = {}
        for position in positions:
            quantity = abs(float(position.qty))
            if position.side is PositionSide.SHORT:
                quantity = -quantity
            value = float(position.market_value) if position.market_value else math.nan
            result[position.symbol] = Position(quantity, value)
        return result

    def open_orders(self) -> list[OpenOrder]:
        """Retourne les ordres encore ouverts."""
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        orders = with_backoff(lambda: self._client.get_orders(request), self._retry)
        if isinstance(orders, dict):
            raise BrokerError("Réponse brute reçue au lieu des ordres")
        result = []
        for order in orders:
            remaining = _to_float(order.qty) - _to_float(order.filled_qty)
            sign = -1.0 if order.side is AlpacaOrderSide.SELL else 1.0
            result.append(OpenOrder(order.symbol or "", sign * remaining))
        return result

    def portfolio(
        self, account: AccountSnapshot, prices: Mapping[str, float]
    ) -> tuple[PortfolioState, set[str]]:
        """Retourne l'état complet du compte et les symboles ayant un ordre ouvert.

        Sans prix pour valoriser un ordre ouvert, sa valeur reste NaN : le module de risque refuse
        alors tout ordre plutôt que de sous-estimer l'exposition.
        """
        positions = self.positions()
        open_orders = self.open_orders()
        for order in open_orders:
            pending = order.signed_quantity * prices.get(order.symbol, math.nan)
            current = positions.get(order.symbol, Position(0.0, 0.0))
            positions[order.symbol] = Position(
                current.quantity, current.market_value, current.pending_notional + pending
            )
        state = PortfolioState(float(account.equity), float(account.last_equity), positions)
        return state, {order.symbol for order in open_orders}

    def submit(
        self, symbol: str, side: OrderSide, quantity: float, client_order_id: str
    ) -> OrderReport:
        """Envoie un ordre au marché, valable pour la journée.

        Un refus du broker ne lève pas d'exception : il est retourné pour être journalisé. Un
        identifiant déjà utilisé retourne l'ordre existant. Une panne réseau lève une exception :
        le sort de l'ordre est alors inconnu, et se retrouve ensuite avec `find_order`.
        """
        request = MarketOrderRequest(
            symbol=symbol,
            qty=quantity,
            side=AlpacaOrderSide(side.value),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            order = with_backoff(lambda: self._client.submit_order(request), self._retry)
        except APIError as exc:
            status = exc.status_code
            if status is None or status >= HTTPStatus.INTERNAL_SERVER_ERROR:
                raise
            existing = self.find_order(client_order_id)
            if existing is not None:
                return existing
            return OrderReport(
                client_order_id=client_order_id,
                broker_order_id=None,
                status=REJECTED_BY_BROKER,
                is_final=True,
                detail=f"HTTP {status}",
            )
        if not isinstance(order, Order):
            raise BrokerError("Réponse brute reçue au lieu de l'ordre")
        return _report(order)

    def find_order(self, client_order_id: str) -> OrderReport | None:
        """Retourne l'état d'un ordre, ou None si le broker ne le connaît pas."""
        try:
            order = with_backoff(
                lambda: self._client.get_order_by_client_id(client_order_id), self._retry
            )
        except APIError as exc:
            if exc.status_code == HTTPStatus.NOT_FOUND:
                return None
            raise
        if not isinstance(order, Order):
            raise BrokerError("Réponse brute reçue au lieu de l'ordre")
        return _report(order)

    def position_details(self) -> list[PositionDetail]:
        """Retourne les positions avec leur prix de revient et leur résultat latent."""
        positions = with_backoff(self._client.get_all_positions, self._retry)
        if isinstance(positions, dict):
            raise BrokerError("Réponse brute reçue au lieu des positions")
        details = []
        for position in positions:
            quantity = abs(float(position.qty))
            if position.side is PositionSide.SHORT:
                quantity = -quantity
            details.append(
                PositionDetail(
                    symbol=position.symbol,
                    quantity=quantity,
                    avg_entry_price=float(position.avg_entry_price),
                    current_price=_or_nan(position.current_price),
                    market_value=_or_nan(position.market_value),
                    unrealized_pl=_to_float(position.unrealized_pl),
                    unrealized_plpc=_to_float(position.unrealized_plpc),
                    change_today=_to_float(position.change_today),
                )
            )
        return sorted(details, key=lambda detail: detail.symbol)

    def portfolio_history(self, request: GetPortfolioHistoryRequest) -> list[EquitySample]:
        """Retourne l'historique du capital, sans les fenêtres où il est inconnu ou nul.

        Alpaca renvoie 0 pour les jours qui précèdent le financement du compte.
        """
        history = with_backoff(lambda: self._client.get_portfolio_history(request), self._retry)
        if not isinstance(history, PortfolioHistory):
            raise BrokerError("Réponse brute reçue au lieu de l'historique du capital")
        return [
            EquitySample(datetime.fromtimestamp(timestamp, UTC), float(equity))
            for timestamp, equity in zip(history.timestamp, history.equity, strict=True)
            if math.isfinite(equity) and equity > 0
        ]

    def cancel_all(self) -> None:
        """Demande l'annulation de tous les ordres ouverts."""
        with_backoff(self._client.cancel_orders, self._retry)
