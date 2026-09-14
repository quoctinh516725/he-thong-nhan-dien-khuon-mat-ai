from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config.env import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_size=10,                    # Tăng connection pool
    max_overflow=20,                 # Cho phép tạo thêm connections
    pool_pre_ping=True,             # Kiểm tra connection trước khi dùng
    echo=False,                      # Tắt log SQL (tắt để tăng tốc)
    connect_args={"connect_timeout": 10}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base = declarative_base()