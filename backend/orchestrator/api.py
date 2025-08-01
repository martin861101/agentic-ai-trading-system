from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from backend.db.db_session import get_db
from backend.orchestrator.models import Event

# Pydantic model for serializing Event data
class EventOut(BaseModel):
    id: int
    timestamp: datetime
    agent_name: str
    event_type: str
    payload: dict

    class Config:
        orm_mode = True

api_router = APIRouter()

@api_router.get("/health", status_code=200)
def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}

@api_router.get("/events", response_model=List[EventOut])
def get_events(
    agent_name: Optional[str] = None,
    event_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve events from the database with optional filtering.
    """
    query = db.query(Event)
    if agent_name:
        query = query.filter(Event.agent_name == agent_name)
    if event_type:
        query = query.filter(Event.event_type == event_type)

    events = query.offset(skip).limit(limit).all()
    return events
