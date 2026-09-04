from fastapi.testclient import TestClient
import pytest
from server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_chat_endpoint_valid_lookup(client):
    res = client.post("/chat", json={"message": "Where is ORD-1007?"})
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "shipped" in data["response"].lower()
    assert data["tool_called"] == "order_lookup"


def test_chat_endpoint_empty_message(client):
    res = client.post("/chat", json={"message": ""})
    assert res.status_code == 400
