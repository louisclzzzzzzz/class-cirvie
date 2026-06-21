from unittest.mock import patch

import pytest

import main as main_module
import predict_server


@pytest.fixture
def bundles(trained_omv, trained_origine, synthetic_agent_to_service):
    return {
        "OMV": trained_omv,
        "ORIGINE": trained_origine,
        "SERVICE": {"agent_to_service": synthetic_agent_to_service, "config": main_module.CONFIG},
    }


@pytest.fixture
def client(bundles):
    with patch("main.load_inference_bundle", side_effect=lambda key: bundles[key]):
        with predict_server.app.test_client() as test_client:
            yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_endpoint_valid_payload(client):
    response = client.post("/predict", json={
        "description": "Le sociétaire constate une erreur dans la base de données de son contrat.",
        "demandeur": "DUPONT, Jean",
    })
    assert response.status_code == 200
    data = response.get_json()
    assert {"omv", "service", "origine"}.issubset(data.keys())


def test_predict_endpoint_missing_description(client):
    response = client.post("/predict", json={"demandeur": "DUPONT, Jean"})
    assert response.status_code == 400


def test_predict_endpoint_omv_non(client):
    response = client.post("/predict", json={
        "description": "Mot de passe oublié, connexion impossible au poste de travail sans impact contrat.",
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data["service"] is None
    assert data["origine"] is None
