"""Unit tests for analysis routes."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from analysis.models import AnalysisEvent
from auth.models import User
from db.database import Base, get_db
from main import app


# Create test database with thread-safe settings for SQLite
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create test client
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_database():
    """Create tables, scope the get_db override to this test, and clean up after.

    The override is applied/reset per-test (rather than once at import time)
    because `tests/auth/test_routes.py` mutates the same `app.dependency_overrides`
    dict — whichever module is collected last would otherwise "win" and silently
    point every test's requests at the wrong in-memory database.
    """
    Base.metadata.create_all(bind=engine)
    previous_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db
    yield
    if previous_override is not None:
        app.dependency_overrides[get_db] = previous_override
    else:
        app.dependency_overrides.pop(get_db, None)
    Base.metadata.drop_all(bind=engine)


class TestTotalAnalysesRoute:
    """Tests for the total-analyses stats route."""

    def test_total_analyses_is_zero_initially(self):
        """Test that the count starts at zero with no events."""
        response = client.get("/stats/total-analyses")

        assert response.status_code == 200
        assert response.json() == {"total_analyses": 0}

    def test_total_analyses_counts_anonymous_and_registered(self):
        """Test that the count includes both anonymous and registered events."""
        db = TestingSessionLocal()
        user = User(username="testuser", hashed_password="password123")
        db.add(user)
        db.commit()

        db.add_all([
            AnalysisEvent(user_id=user.id),
            AnalysisEvent(),
            AnalysisEvent(),
        ])
        db.commit()
        db.close()

        response = client.get("/stats/total-analyses")

        assert response.status_code == 200
        assert response.json() == {"total_analyses": 3}

    def test_total_analyses_does_not_require_auth(self):
        """Test that the endpoint is publicly accessible without a token."""
        response = client.get("/stats/total-analyses")

        assert response.status_code == 200
