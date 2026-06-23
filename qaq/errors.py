"""Project-level exceptions with stable validation categories."""

from __future__ import annotations

from dataclasses import dataclass


class QaqError(Exception):
    """Base class for QAQ errors that should produce a non-zero command exit."""

    exit_code = 1


@dataclass(slots=True)
class ConfigValidationError(QaqError):
    """Raised when a run configuration fails schema or semantic validation."""

    code: str
    message: str
    field: str | None = None

    exit_code = 2

    def __str__(self) -> str:
        if self.field:
            return f"{self.code}: {self.field}: {self.message}"
        return f"{self.code}: {self.message}"


@dataclass(slots=True)
class ManifestError(QaqError):
    """Raised when a run manifest cannot be created or updated."""

    code: str
    message: str

    exit_code = 3

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"
