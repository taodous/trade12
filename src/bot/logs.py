"""Configuration des logs structurés."""

import structlog


def configure_logging() -> None:
    """Configure structlog pour écrire des logs JSON horodatés en UTC."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ]
    )
