import pandas as pd

import main as main_module


def test_normalize_demandeur_csv_with_comma():
    assert main_module.normalize_demandeur_csv("DUPONT, Jean") == "JEAN DUPONT"


def test_normalize_demandeur_csv_nan():
    assert main_module.normalize_demandeur_csv(float("nan")) == "INCONNU"


def test_clean_fills_inconnu_for_missing_fields():
    df = pd.DataFrame({
        "Description": ["Un incident quelconque sur le contrat."],
        "INTERVENTION OMV": ["OUI"],
        "SERVICE": ["TOUS"],
        "ORIGINE": ["AUTRE"],
    })
    cleaned = main_module.clean(df, {})
    assert cleaned["CAUSE"].iloc[0] == "INCONNU"
    assert cleaned["TRAITE"].iloc[0] == "INCONNU"
    assert cleaned["DEMANDEUR"].iloc[0] == "INCONNU"


def test_ville_extraction():
    df = pd.DataFrame({"Description": ["Sociétaire domicilié à PARIS pour son contrat."]})
    cleaned = main_module.clean(df, {})
    assert cleaned["VILLE"].iloc[0] == "PARIS"


def test_ville_none_when_absent():
    df = pd.DataFrame({"Description": ["Incident sans mention de ville particulière."]})
    cleaned = main_module.clean(df, {})
    assert cleaned["VILLE"].iloc[0] == "NON_PRECISEE"


def test_urgence_cat_from_numeric():
    df = pd.DataFrame({"Description": ["test"], "Urgence": ["2"]})
    cleaned = main_module.clean(df, {})
    assert cleaned["URGENCE_CAT"].iloc[0] == "2"


def test_omv_invalid_values_set_to_nan():
    df = pd.DataFrame({
        "Description": ["test"],
        "INTERVENTION OMV": ["PEUT-ETRE"],
    })
    cleaned = main_module.clean(df, {})
    assert pd.isna(cleaned["OMV"].iloc[0])


def test_filter_classes_removes_rare():
    df = pd.DataFrame({"target": ["A"] * 35 + ["B"] * 2})
    filtered = main_module._filter_classes(df, "target", min_samples=30)
    assert "B" not in filtered["target"].unique()
    assert "A" in filtered["target"].unique()
