import os
from threading import Event, Thread

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.common.handlers import register_common_handlers
from app.services import ai as _ai  # Load torch models before database modules on Apple Silicon.
from app.routers import simplified_faces
from app.database.db import SessionLocal
from app.services.avatar_cache import load_avatar_cache
from app.services.face_index import load_face_embedding_index
from app.services.object_storage import get_image_storage

app = FastAPI()
register_common_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simplified_faces.router)

face_index_refresh_stop = Event()
face_index_refresh_thread = None


def refresh_face_index_periodically(interval_seconds: float):
    while not face_index_refresh_stop.wait(interval_seconds):
        db = SessionLocal()
        try:
            load_face_embedding_index(db)
        except Exception as exc:
            print(f"WARNING: Could not refresh face embedding cache: {exc}")
        finally:
            db.close()

@app.on_event("startup")
def startup_event():
    global face_index_refresh_thread
    db = SessionLocal()
    try:
        get_image_storage().ensure_ready()
        load_avatar_cache(db)
        load_face_embedding_index(db)
    finally:
        db.close()

    refresh_seconds = float(os.getenv("FACE_INDEX_REFRESH_SECONDS", "5"))
    if refresh_seconds > 0:
        face_index_refresh_stop.clear()
        face_index_refresh_thread = Thread(
            target=refresh_face_index_periodically,
            args=(refresh_seconds,),
            daemon=True,
            name="face-index-refresh",
        )
        face_index_refresh_thread.start()


@app.on_event("shutdown")
def shutdown_event():
    face_index_refresh_stop.set()
    if face_index_refresh_thread is not None:
        face_index_refresh_thread.join(timeout=1.0)

@app.get("/")
def root():
    return {"message": "Hello World"}
