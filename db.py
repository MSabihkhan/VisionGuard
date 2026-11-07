import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import DateTime, Float, Integer, String, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///history.db")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    ai_score: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))


def init_db():
    Base.metadata.create_all(engine)


def add_prediction(filename, label, confidence, ai_score, model_name):
    with Session(engine) as session:
        session.add(Prediction(filename=filename, label=label, confidence=confidence, ai_score=ai_score, model_name=model_name))
        session.commit()


def recent(limit=100):
    with Session(engine) as session:
        return list(session.scalars(select(Prediction).order_by(Prediction.id.desc()).limit(limit)))


def ping():
    with Session(engine) as session:
        session.execute(text("SELECT 1"))
