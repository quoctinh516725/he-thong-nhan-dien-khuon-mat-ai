import secrets

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.common.exceptions import AppError
from app.common.response import error_response
from app.services.image_input import ImageInputError
from app.vision.exceptions import AmbiguousFaceError, NoFaceDetectedError, VisionError


class TraceIdMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        trace_id = secrets.token_urlsafe(18)
        scope.setdefault("state", {})["trace_id"] = trace_id

        async def send_with_trace(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-trace-id", trace_id.encode()))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_trace)


def register_common_handlers(app: FastAPI) -> None:
    app.add_middleware(TraceIdMiddleware)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError):
        return _json_error(request, exc.status_code, exc.error_code, exc.message)

    @app.exception_handler(ImageInputError)
    async def handle_image_input_error(request: Request, exc: ImageInputError):
        return _json_error(request, 400, "INVALID_IMAGE_INPUT", str(exc))

    @app.exception_handler(AmbiguousFaceError)
    async def handle_ambiguous_face_error(request: Request, exc: AmbiguousFaceError):
        return _json_error(
            request,
            400,
            "AMBIGUOUS_FACE",
            "Phát hiện nhiều gương mặt tương đương nhau. Vui lòng gửi ảnh có một khuôn mặt chính rõ ràng.",
        )

    @app.exception_handler(NoFaceDetectedError)
    async def handle_no_face_error(request: Request, exc: NoFaceDetectedError):
        return _json_error(request, 400, "NO_FACE_DETECTED", "Không phát hiện khuôn mặt trong ảnh.")

    @app.exception_handler(VisionError)
    async def handle_vision_error(request: Request, exc: VisionError):
        return _json_error(request, 400, "VISION_ERROR", str(exc))

    @app.exception_handler(SQLAlchemyError)
    async def handle_database_error(request: Request, exc: SQLAlchemyError):
        import traceback
        traceback.print_exc()
        return _json_error(request, 500, "DATABASE_ERROR", "Không xử lý được truy vấn dữ liệu.")

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return _json_error(request, 422, "VALIDATION_ERROR", "Dữ liệu request không hợp lệ.")

    @app.exception_handler(Exception)
    async def handle_generic_error(request: Request, exc: Exception):
        import traceback
        traceback.print_exc()
        return _json_error(request, 500, "INTERNAL_SERVER_ERROR", str(exc))



def _json_error(request: Request, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=error_response(request, status_code, code, message).model_dump(),
    )
