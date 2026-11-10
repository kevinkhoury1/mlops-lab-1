from fastapi.testclient import TestClient

from src.food11 import serve


class FakeModel:
    def predict(self, model_input):
        assert model_input.shape == (1, 3, 128, 128)
        return [[0.90] + [0.01] * 10]


def test_health(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_503_without_model(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with (
        TestClient(serve.app) as client,
        open("tests/fixtures/bread.jpg", "rb") as image,
    ):
        response = client.post(
            "/predict", files={"file": ("bread.jpg", image, "image/jpeg")}
        )
    assert response.status_code == 503


def test_predict_returns_valid_json(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL", "1")
    with TestClient(serve.app) as client:
        monkeypatch.setattr(serve, "model", FakeModel())
        with open("tests/fixtures/bread.jpg", "rb") as image:
            response = client.post(
                "/predict", files={"file": ("bread.jpg", image, "image/jpeg")}
            )

    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "Bread"
    assert 0.0 <= body["confidence"] <= 1.0
