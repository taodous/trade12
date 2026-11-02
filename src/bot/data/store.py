"""Stockage local des barres, reproductible et vérifié par empreinte."""

import hashlib
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from bot.data.quality import DataIssue


class StoreError(RuntimeError):
    """Jeu de données absent, corrompu ou déjà présent."""


class BarMetadata(BaseModel):
    """Provenance d'un jeu de barres, et anomalies détectées à son téléchargement."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    feed: str
    adjustment: str
    start: date
    end: date
    fetched_at: datetime
    rows: int
    sha256: str
    issues: list[DataIssue]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BarStore:
    """Un fichier Parquet et un fichier de provenance JSON par symbole.

    Les prix ajustés changent rétroactivement à chaque dividende. Un jeu existant n'est donc jamais
    remplacé sans demande explicite, pour qu'un backtest reste reproductible.
    """

    def __init__(self, directory: Path) -> None:
        """Associe le stockage à `directory`, créé à la première écriture."""
        self._directory = directory

    def _bars_path(self, symbol: str) -> Path:
        return self._directory / f"{symbol}.parquet"

    def _metadata_path(self, symbol: str) -> Path:
        return self._directory / f"{symbol}.json"

    def exists(self, symbol: str) -> bool:
        """Indique si des barres sont stockées pour `symbol`."""
        return self._bars_path(symbol).exists() or self._metadata_path(symbol).exists()

    def save(
        self,
        symbol: str,
        bars: pd.DataFrame,
        *,
        feed: str,
        adjustment: str,
        fetched_at: datetime,
        issues: list[DataIssue],
        overwrite: bool = False,
    ) -> BarMetadata:
        """Écrit les barres et leur provenance, puis retourne cette provenance."""
        if bars.empty:
            raise StoreError(f"Aucune barre à stocker pour {symbol}")
        if self.exists(symbol) and not overwrite:
            raise StoreError(f"Des barres existent déjà pour {symbol}")

        self._directory.mkdir(parents=True, exist_ok=True)
        bars_path = self._bars_path(symbol)
        temporary = bars_path.with_suffix(".parquet.tmp")
        bars.to_parquet(temporary)
        metadata = BarMetadata(
            symbol=symbol,
            feed=feed,
            adjustment=adjustment,
            start=pd.Timestamp(str(bars.index.min())).date(),
            end=pd.Timestamp(str(bars.index.max())).date(),
            fetched_at=fetched_at,
            rows=len(bars),
            sha256=_sha256(temporary),
            issues=issues,
        )
        temporary.replace(bars_path)
        self._metadata_path(symbol).write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        return metadata

    def load_metadata(self, symbol: str) -> BarMetadata:
        """Retourne la provenance des barres de `symbol`."""
        try:
            text = self._metadata_path(symbol).read_text(encoding="utf-8")
            return BarMetadata.model_validate_json(text)
        except (OSError, ValidationError) as exc:
            raise StoreError(f"Provenance illisible pour {symbol} : {exc}") from exc

    def load(self, symbol: str) -> pd.DataFrame:
        """Retourne les barres de `symbol`, après vérification de leur empreinte."""
        metadata = self.load_metadata(symbol)
        bars_path = self._bars_path(symbol)
        try:
            digest = _sha256(bars_path)
        except OSError as exc:
            raise StoreError(f"Barres illisibles pour {symbol} : {exc}") from exc
        if digest != metadata.sha256:
            raise StoreError(f"Empreinte incorrecte pour {symbol} : fichier modifié ou corrompu")
        return pd.read_parquet(bars_path)

    def load_closes(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Retourne les clôtures, une ligne par séance et une colonne par symbole.

        Une séance absente pour un symbole reste vide : aucune valeur n'est reportée.
        """
        if not symbols:
            raise StoreError("Aucun symbole demandé")
        closes = {symbol: self.load(symbol)["close"] for symbol in symbols}
        return pd.DataFrame(closes).sort_index()
