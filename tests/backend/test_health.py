"""Unit tests for FastAPI health endpoint."""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verifies root endpoint returns project metadata and disclaimer."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "NeuroSurgical AI" in data["project"]
    assert "disclaimer" in data
    assert "Academic decision-support" in data["disclaimer"]


def test_health_endpoint():
    """Verifies health check endpoint returns 200 with cpu device."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["device"] == "cpu"
    assert "disclaimer" in data
