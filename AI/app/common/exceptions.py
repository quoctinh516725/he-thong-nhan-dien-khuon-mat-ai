from http import HTTPStatus
from typing import Optional


class AppError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_code: Optional[str] = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or HTTPStatus(status_code).phrase.upper().replace(" ", "_")
        super().__init__(message)
