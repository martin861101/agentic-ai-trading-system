from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from backend.db.db_session import Base

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    agent_name = Column(String, index=True)
    event_type = Column(String, index=True)
    payload = Column(JSON)

    def __repr__(self):
        return f"<Event(id={self.id}, agent='{self.agent_name}', type='{self.event_type}')>"
