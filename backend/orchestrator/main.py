import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.db.db_session import engine, Base
from backend.orchestrator.api import api_router
from backend.config import settings

# Configure logging
logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

def create_db_and_tables():
    """
    Creates the database and all tables defined in the Base metadata.
    """
    try:
        logger.info("Creating database and tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database and tables created successfully.")
    except Exception as e:
        logger.error(f"Error creating database and tables: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for the FastAPI application.
    """
    logger.info("Starting up...")
    if settings.ENV != "test":
        logger.info("Running in non-test environment, creating database tables.")
        create_db_and_tables()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Agentic Trading Platform Orchestrator",
    description="Coordinates the activities of all specialized agents.",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    """
    Root endpoint providing a simple status message.
    """
    return {"message": "Orchestrator is running"}
