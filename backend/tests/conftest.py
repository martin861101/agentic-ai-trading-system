import os
os.environ['ENV'] = 'test'

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def engine():
    # Import Base and models inside the fixture to ensure they are loaded
    # at the time the fixture is executed.
    from backend.db.db_session import Base
    from backend.orchestrator import models

    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine

@pytest.fixture(scope="function")
def db_session(engine):
    """
    Creates a new database session for each test function.
    """
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture(scope="function")
def client(db_session):
    """
    Creates a TestClient for the FastAPI app, with the database dependency overridden.
    """
    # Import app and dependency inside the fixture
    from backend.orchestrator.main import app
    from backend.db.db_session import get_db

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    # Clean up the dependency override after the test
    app.dependency_overrides.clear()
