import numpy as np

import main as main_module


def test_omv_accuracy_above_threshold(trained_omv):
    assert trained_omv["metrics"]["accuracy"] >= 0.80


def test_origine_f1_above_threshold(trained_origine):
    assert trained_origine["metrics"]["f1_macro"] >= 0.50


def test_calibrated_probabilities_sum_to_one(trained_omv, trained_origine):
    for result in (trained_omv, trained_origine):
        sums = result["y_prob"].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5)


def test_confidence_scores_in_range(patched_inference_cache):
    result = main_module.predict(
        description="Le sociétaire constate une erreur de saisie sur le remboursement de son contrat.",
        demandeur="DUPONT, Jean",
    )
    for key in ("omv", "service", "origine"):
        if result[key] is not None:
            assert 0.0 <= result[key]["confidence"] <= 100.0


def test_prediction_is_deterministic(patched_inference_cache):
    kwargs = dict(
        description="Erreur de saisie manuelle lors de l'enregistrement du dossier contrat.",
        demandeur="MARTIN, Marie",
    )
    first = main_module.predict(**kwargs)
    second = main_module.predict(**kwargs)
    assert first == second


def test_omv_non_skips_service_and_origine(patched_inference_cache):
    result = main_module.predict(
        description="Mot de passe oublié, connexion impossible au poste de travail sans impact contrat.",
    )
    assert result["omv"]["prediction"] == "NON"
    assert result["service"] is None
    assert result["origine"] is None


def test_omv_oui_returns_all_three_labels(patched_inference_cache):
    result = main_module.predict(
        description="Le sociétaire constate une erreur dans la base de données de son contrat.",
        demandeur="DUPONT, Jean",
    )
    assert result["omv"]["prediction"] == "OUI"
    assert result["service"] is not None
    assert result["origine"] is not None


def test_low_confidence_flagged(monkeypatch, patched_inference_cache):
    def fake_infer_single(key, row, top_n):
        if key == "OMV":
            return {
                "prediction": "OUI",
                "confidence": 0.3,
                "top_n": [{"classe": "OUI", "probabilite": 0.3}, {"classe": "NON", "probabilite": 0.7}],
            }
        return {
            "prediction": "BASE DE DONNÉES",
            "confidence": 0.3,
            "top_n": [{"classe": "BASE DE DONNÉES", "probabilite": 0.3}],
        }

    monkeypatch.setattr(main_module, "_infer_single", fake_infer_single)
    result = main_module.predict(description="test", confidence_threshold=0.5)
    assert result["omv"]["flagged_for_review"] is True
    assert result["omv"]["reason"] == "low_confidence"
    assert result["origine"]["flagged_for_review"] is True


def test_high_confidence_not_flagged(monkeypatch, patched_inference_cache):
    def fake_infer_single(key, row, top_n):
        if key == "OMV":
            return {
                "prediction": "OUI",
                "confidence": 0.9,
                "top_n": [{"classe": "OUI", "probabilite": 0.9}, {"classe": "NON", "probabilite": 0.1}],
            }
        return {
            "prediction": "BASE DE DONNÉES",
            "confidence": 0.9,
            "top_n": [{"classe": "BASE DE DONNÉES", "probabilite": 0.9}],
        }

    monkeypatch.setattr(main_module, "_infer_single", fake_infer_single)
    result = main_module.predict(description="test", confidence_threshold=0.5)
    assert not result["omv"].get("flagged_for_review", False)
    assert not result["origine"].get("flagged_for_review", False)
