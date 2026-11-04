"""Création du client Alpaca, limitée au mode paper, et lecture du compte."""

from decimal import Decimal
from typing import Protocol

from alpaca.common.enums import BaseURL
from alpaca.common.types import RawData
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AccountStatus
from alpaca.trading.models import TradeAccount
from pydantic import BaseModel, ConfigDict, ValidationError

from bot.config import Settings


class PaperModeError(RuntimeError):
    """Tentative de création d'un client qui ne vise pas le compte paper."""


class AccountDataError(RuntimeError):
    """Réponse du compte inexploitable."""


class AccountClient(Protocol):
    """Partie de l'API Alpaca nécessaire à la lecture du compte."""

    def get_account(self) -> TradeAccount | RawData:
        """Retourne le compte."""
        ...


class AccountSnapshot(BaseModel):
    """État du compte, sans son identifiant ni son numéro."""

    model_config = ConfigDict(frozen=True, from_attributes=True, hide_input_in_errors=True)

    status: AccountStatus
    currency: str
    equity: Decimal
    last_equity: Decimal
    cash: Decimal
    buying_power: Decimal
    trading_blocked: bool
    account_blocked: bool

    @property
    def can_trade(self) -> bool:
        """Indique si le compte est actif et sans blocage."""
        return (
            self.status is AccountStatus.ACTIVE
            and not self.trading_blocked
            and not self.account_blocked
        )


def create_trading_client(settings: Settings, *, paper: bool = True) -> TradingClient:
    """Crée le client de trading paper.

    Lève `PaperModeError` si le mode paper est désactivé ou si le client obtenu ne vise pas
    l'endpoint paper.
    """
    if paper is not True:
        raise PaperModeError("Seul le paper trading est autorisé")

    client = TradingClient(
        api_key=settings.alpaca_api_key.get_secret_value(),
        secret_key=settings.alpaca_secret_key.get_secret_value(),
        paper=True,
    )
    if client._base_url != BaseURL.TRADING_PAPER:
        raise PaperModeError("Le client ne vise pas l'endpoint paper d'Alpaca")
    return client


def read_account(client: AccountClient) -> AccountSnapshot:
    """Lit le compte, ou lève `AccountDataError` si la réponse est inexploitable."""
    account = client.get_account()
    if not isinstance(account, TradeAccount):
        raise AccountDataError("Réponse brute reçue au lieu d'un TradeAccount")
    try:
        return AccountSnapshot.model_validate(account)
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors())
        raise AccountDataError(f"Champs du compte invalides : {fields}") from None
