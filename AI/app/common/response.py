from typing import Generic, Optional, TypeVar

from fastapi import Request
from pydantic import BaseModel


T = TypeVar("T")


class ErrorVO(BaseModel):
    code: str
    message: str


class ResponseVO(BaseModel, Generic[T]):
    traceId: str
    status: str
    result: str
    error: Optional[ErrorVO] = None
    data: Optional[T] = None


def success_response(request: Request, data: T, status_code: int = 200) -> ResponseVO[T]:
    return ResponseVO(
        traceId=get_trace_id(request),
        status=str(status_code),
        result="Succeeded",
        error=None,
        data=data,
    )


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
) -> ResponseVO[None]:
    return ResponseVO(
        traceId=get_trace_id(request),
        status=str(status_code),
        result="Failed",
        error=ErrorVO(code=code, message=message),
        data=None,
    )


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "")
