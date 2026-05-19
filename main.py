"""
Classification CIRVIE rapide sans CamemBERT.

Le script conserve les idées clés du pipeline principal historique :
  1. trois modèles indépendants (OMV / SERVICE / ORIGINE)
  2. enrichissement par variables métier tabulaires
  3. règle déterministe prioritaire pour SERVICE via nom_service.json
  4. fonction d'inférence réutilisable predict(...)

Le coeur modèle est remplacé par un pipeline scikit-learn sparse :
  - TF-IDF sur Description
  - OneHotEncoder sur les variables catégorielles
  - LogisticRegression
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import LinearSVC

from metrics_logger import log_results

warnings.filterwarnings("ignore")


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "min_samples": 30,
    "tfidf": {
        "OMV": {
            "max_features": 8000,
            "ngram_range": (2, 3),
            "min_df": 2,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
        "ORIGINE": {
            "max_features": 20000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        },
    },
    "linearsvc": {
        "OMV": {"C": 0.5},
        "ORIGINE": {"C": 5.0},
    },
}

def update_config_from_args(args: argparse.Namespace):
    if args.test_size:
        CONFIG["test_size"] = args.test_size
    if args.max_features:
        CONFIG["tfidf"]["OMV"]["max_features"] = args.max_features
        CONFIG["tfidf"]["ORIGINE"]["max_features"] = args.max_features
    if args.c_omv:
        CONFIG["linearsvc"]["OMV"]["C"] = args.c_omv
    if args.c_origine:
        CONFIG["linearsvc"]["ORIGINE"]["C"] = args.c_origine

DATA_FILE = "data3.csv"
NOM_SERVICE_FILE = "nom_service.json"

TARGET_OMV = "INTERVENTION OMV"
TARGET_SERVICE = "SERVICE"
TARGET_ORIGINE = "ORIGINE"

BASELINE_METRICS = {
    "OMV": {"accuracy": 0.985, "f1_macro": 0.990},
    "SERVICE": {"accuracy": 0.758, "f1_macro": 0.665},
    "ORIGINE": {"accuracy": 0.735, "f1_macro": 0.752},
}

ORIGINE_MAP = {
    "BASE DE DONNEES": "BASE DE DONNÉES",
    "ASSISTANCE UTILISATEUR": "ASSISTANCE UTILISATEURS",
    "PARAMETRAGE": "PARAMÉTRAGE",
    "ERREUR SAISIE": "ERREUR DE SAISIE",
    "TELEPHONE": "TÉLÉPHONE",
    "DONNEES": "DONNÉES",
    "FISCALITE": "FISCALITÉ",
}

SERVICE_MAP = {
    "DÉCÈS": "DECES",
}

VILLES = [
    "PARIS", "NANCY", "LYON", "BORDEAUX", "MARSEILLE",
    "TOULOUSE", "NANTES", "STRASBOURG", "LILLE", "RENNES",
]
VILLE_PATTERN = r"\b(" + "|".join(VILLES) + r")\b"

TABULAR_CONFIG = {
    "OMV": ["CAUSE", "TRAITE", "DEMANDEUR", "VILLE", "URGENCE_CAT"],
    "ORIGINE": ["CAUSE", "TRAITE", "DEMANDEUR", "VILLE", "URGENCE_CAT"],
}

ARTIFACTS = {
    "OMV": "model_fast_omv.pkl",
    "SERVICE": "model_fast_service.pkl",
    "ORIGINE": "model_fast_origine.pkl",
}

TUNING_GRIDS: dict[str, dict[str, Any]] = {
    "logreg": {
        "label": "LogisticRegression",
        "grids": {
            "OMV":    {"C": [0.5, 1.0, 2.0, 5.0], "solver": ["liblinear", "saga"], "max_iter": [1500]},
            "ORIGINE": {"C": [1.0, 2.0, 5.0, 10.0], "solver": ["saga"], "max_iter": [2500]},
        },
    },
    "linearsvc": {
        "label": "LinearSVC (calibré)",
        "grids": {
            "OMV":    {"C": [0.1, 0.5, 1.0, 5.0]},
            "ORIGINE": {"C": [0.1, 0.5, 1.0, 5.0]},
        },
    },
    "random_forest": {
        "label": "RandomForest",
        "grids": {
            "OMV":    {"n_estimators": [100, 200], "max_depth": [None, 10], "min_samples_leaf": [1, 3]},
            "ORIGINE": {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]},
        },
    },
}

MANUAL_TEST_CASES = [
    {
        "label": "Rachat contrat individuel Paris",
        "description": (
            "Rachat partiel sur le contrat 30421. Situation attendue : courrier envoyé "
            "au sociétaire. Situation obtenue : absence de courrier. "
            "Sociétaire domicilié à Paris."
        ),
        "demandeur": "TASSINE-BELLOUT, Anne",
        "cause": "BASE DE DONNÉES",
        "traite": "MOA OMVIE",
        "expected_omv": "OUI",
        "expected_service": "SER IND PARIS",
        "expected_origine": "BASE DE DONNÉES",
    },
    {
        "label": "Mot de passe oublié",
        "description": (
            "Blocage connexion CIRVIE PF5. L'utilisateur a oublié son mot de passe "
            "et ne peut plus se connecter à l'application depuis ce matin."
        ),
        "demandeur": "INCONNU",
        "cause": "MOT DE PASSE",
        "traite": "EXP SUPPORT HABILITATION",
        "expected_omv": "NON",
        "expected_service": None,
        "expected_origine": None,
    },
    {
        "label": "Anomalie 990I décès",
        "description": (
            "INCIDENT CIRVIE. Situation attendue : capitaux réglés corrects avec "
            "abattement appliqué une seule fois. Situation obtenue : abattement "
            "990I appliqué plusieurs fois sur les 4 contrats du sociétaire décédé."
        ),
        "demandeur": "MIDETON, Elodie",
        "cause": "PROGRAMME",
        "traite": "MOA OMVIE",
        "expected_omv": "OUI",
        "expected_service": "DECES",
        "expected_origine": "PROGRAMME",
    },
    {
        "label": "Job batch en échec",
        "description": (
            "Le job A1P1CV45FA est en état JOBFAILURE sur SMASPRO1. "
            "Traitement batch nocturne en échec, relance nécessaire."
        ),
        "demandeur": "INCONNU",
        "cause": "BATCH",
        "traite": "EXP OPÉRATIONS DOMAINE VCP",
        "expected_omv": "NON",
        "expected_service": None,
        "expected_origine": None,
    },
    {
        "label": "Sinistre collectif Nancy",
        "description": (
            "Sinistre déclaré par un sociétaire de Nancy. Situation obtenue : "
            "le sinistre n'apparaît pas dans CIRVIE malgré la saisie. "
            "Contrat collectif, référence 78421."
        ),
        "demandeur": "COLLOT, Amandine",
        "cause": "BASE DE DONNÉES",
        "traite": "ETD DOMAINE VIE",
        "expected_omv": "OUI",
        "expected_service": "SER COLL NANCY",
        "expected_origine": "BASE DE DONNÉES",
    },
]

_INFERENCE_CACHE: dict[str, dict[str, Any]] = {}


# ============================================================================
# 2. AFFICHAGE
# ============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def section(title: str) -> None:
    print(f"\n-- {title}")


def metric_line(name: str, value: str) -> None:
    print(f"  {name:<28} {value}")


# ============================================================================
# 3. CHARGEMENT & NETTOYAGE
# ============================================================================

def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def _normalize_agent_name(raw: str) -> str:
    return re.sub(r"[*#]+", "", raw).strip().upper()


def normalize_demandeur_csv(raw: str | float) -> str:
    if pd.isna(raw) or str(raw).strip() == "":
        return "INCONNU"
    value = str(raw).strip()
    if "," in value:
        last, first = value.split(",", 1)
        return f"{first.strip()} {last.strip()}".upper()
    return value.upper()


def build_agent_to_service(json_path: str | Path) -> dict[str, str]:
    with open(json_path, encoding="utf-8") as handle:
        data = json.load(handle)

    agent_to_service: dict[str, str] = {}
    for service_key, raw_value in data.items():
        service_label = service_key.replace("_", " ")

        members: list[str] = []
        if isinstance(raw_value, dict):
            members = raw_value.get("membres", [])
            if "responsable" in raw_value:
                members = members + [raw_value["responsable"]]
        elif isinstance(raw_value, list):
            members = raw_value

        for agent in members:
            normalized = _normalize_agent_name(agent)
            if not normalized:
                continue
            agent_to_service[normalized] = service_label
            agent_to_service[_strip_accents(normalized)] = service_label

    return agent_to_service


def lookup_service(demandeur_normalized: str, agent_to_service: dict[str, str]) -> str | None:
    if demandeur_normalized == "INCONNU":
        return None
    service = agent_to_service.get(demandeur_normalized)
    if service is None:
        service = agent_to_service.get(_strip_accents(demandeur_normalized))
    return service


def normalize_text_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("_x000D_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NAN": np.nan, "NONE": np.nan})
    )


def normalize_free_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace("_x000D_", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_all_sources() -> pd.DataFrame:
    path = Path(DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {DATA_FILE}\n"
            "Placez data3.csv dans le répertoire courant avant de lancer le script."
        )
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    print(f"  chargé : {DATA_FILE:<28} ({len(df)} lignes, {df.shape[1]} colonnes)")
    return df


def clean(df: pd.DataFrame, agent_to_service: dict[str, str]) -> pd.DataFrame:
    df = df.copy()

    df["Description"] = normalize_free_text(df.get("Description", pd.Series([""] * len(df))))
    df["OMV"] = normalize_text_series(df.get(TARGET_OMV, pd.Series([np.nan] * len(df))))
    df["SERVICE_clean"] = normalize_text_series(
        df.get(TARGET_SERVICE, pd.Series([np.nan] * len(df)))
    ).replace(SERVICE_MAP)
    df["ORIGINE_clean"] = normalize_text_series(
        df.get(TARGET_ORIGINE, pd.Series([np.nan] * len(df)))
    ).replace(ORIGINE_MAP)

    df["CAUSE"] = normalize_text_series(
        df.get("Cause réelle", pd.Series([np.nan] * len(df)))
    ).fillna("INCONNU")
    df["TRAITE"] = normalize_text_series(
        df.get("Traité par", pd.Series([np.nan] * len(df)))
    ).fillna("INCONNU")

    raw_demandeur = df.get("Demandeur", pd.Series([np.nan] * len(df)))
    df["DEMANDEUR"] = raw_demandeur.apply(normalize_demandeur_csv)
    df["SERVICE_DETERMINISTE"] = df["DEMANDEUR"].apply(
        lambda demandeur: lookup_service(demandeur, agent_to_service)
    )
    df["VILLE"] = (
        df["Description"].str.upper().str.extract(VILLE_PATTERN, expand=False).fillna("NON_PRECISEE")
    )

    urgence_raw = normalize_text_series(df.get("Urgence", pd.Series([np.nan] * len(df))))
    df["URGENCE_NUM"] = (
        pd.to_numeric(urgence_raw.str.extract(r"(\d)", expand=False), errors="coerce").fillna(-1.0)
    )
    df["URGENCE_CAT"] = np.where(df["URGENCE_NUM"] >= 0, df["URGENCE_NUM"].astype(int).astype(str), "INCONNU")

    df.loc[~df["OMV"].isin(["OUI", "NON"]), "OMV"] = np.nan
    return df


# ============================================================================
# 4. FEATURES
# ============================================================================

def _make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def fit_feature_pipeline(
    train_df: pd.DataFrame,
    cat_cols: list[str],
    tfidf_config: dict[str, Any],
) -> tuple[TfidfVectorizer, OneHotEncoder]:
    tfidf = TfidfVectorizer(**tfidf_config)
    tfidf.fit(train_df["Description"])

    ohe = _make_ohe()
    ohe.fit(train_df[cat_cols].fillna("INCONNU").astype(str))
    return tfidf, ohe


def transform_features(
    df: pd.DataFrame,
    tfidf: TfidfVectorizer,
    ohe: OneHotEncoder,
    cat_cols: list[str],
) -> sp.csr_matrix:
    x_text = tfidf.transform(df["Description"])
    x_cat = ohe.transform(df[cat_cols].fillna("INCONNU").astype(str))
    return sp.hstack([x_text, x_cat], format="csr")


def build_model(target_key: str) -> CalibratedClassifierCV:
    params = CONFIG["linearsvc"][target_key]
    return CalibratedClassifierCV(
        LinearSVC(
            C=params["C"],
            class_weight="balanced",
            max_iter=3000,
            dual="auto",
            random_state=CONFIG["random_state"],
        )
    )


# ============================================================================
# 5. ENTRAÎNEMENT & ÉVALUATION
# ============================================================================

def _filter_classes(df: pd.DataFrame, target_col: str, min_samples: int) -> pd.DataFrame:
    counts = df[target_col].value_counts()
    valid = counts[counts >= min_samples].index
    removed = int((~df[target_col].isin(valid)).sum())
    kept = df[df[target_col].isin(valid)].copy()
    section("Données")
    metric_line("Lignes utiles", str(len(kept)))
    metric_line("Classes retenues", str(kept[target_col].nunique()))
    metric_line("Lignes ignorées (<min_samples)", str(removed))
    return kept


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    if y_prob.shape[1] == 2:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob[:, 1])
    return metrics



def _train_and_eval(
    df_sub: pd.DataFrame,
    target_col: str,
    target_key: str,
    cat_cols: list[str],
) -> dict[str, Any]:
    label_encoder = LabelEncoder()
    df_sub = df_sub.copy()
    df_sub["target_encoded"] = label_encoder.fit_transform(df_sub[target_col])

    stratify = (
        df_sub["target_encoded"]
        if df_sub["target_encoded"].value_counts().min() >= 2
        else None
    )
    df_train, df_test = train_test_split(
        df_sub,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=stratify,
    )

    tfidf, ohe = fit_feature_pipeline(df_train, cat_cols, CONFIG["tfidf"][target_key])
    x_train = transform_features(df_train, tfidf, ohe, cat_cols)
    x_test = transform_features(df_test, tfidf, ohe, cat_cols)

    y_train = df_train["target_encoded"].to_numpy(dtype=np.int64)
    y_test = df_test["target_encoded"].to_numpy(dtype=np.int64)

    model = build_model(target_key)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    metrics = evaluate_predictions(y_test, y_pred, y_prob)

    return {
        "model": model,
        "tfidf": tfidf,
        "ohe": ohe,
        "label_encoder": label_encoder,
        "cat_cols": cat_cols,
        "target_col": target_col,
        "metrics": metrics,
        "y_true_encoded": y_test,
        "y_pred_encoded": y_pred,
        "y_prob": y_prob,
        "y_true": label_encoder.inverse_transform(y_test),
        "y_pred": label_encoder.inverse_transform(y_pred),
        "test_dataframe": df_test.reset_index(drop=True),
    }


def _print_test_metrics(result: dict[str, Any], section_name: str) -> None:
    section(section_name)
    metrics = result["metrics"]
    metric_line("Accuracy", f"{metrics['accuracy']:.4f}")
    metric_line("F1 macro", f"{metrics['f1_macro']:.4f}")
    metric_line("F1 weighted", f"{metrics['f1_weighted']:.4f}")
    if "auc_roc" in metrics:
        metric_line("AUC-ROC", f"{metrics['auc_roc']:.4f}")
    print("\nRapport par classe :")
    print(classification_report(result["y_true"], result["y_pred"], zero_division=0))


# ============================================================================
# 6. PIPELINES PAR CIBLE
# ============================================================================

def run_pipeline_omv(df: pd.DataFrame) -> dict[str, Any]:
    banner("ETAPE 1 — OMV (OUI / NON)")
    df_sub = df[df["OMV"].notna() & (df["Description"] != "")].copy()
    df_sub = _filter_classes(df_sub, "OMV", CONFIG["min_samples"])
    result = _train_and_eval(df_sub, "OMV", "OMV", TABULAR_CONFIG["OMV"])
    _print_test_metrics(result, "Résultats test — OMV")
    result["label"] = "OMV"
    return result


def run_pipeline_origine(df: pd.DataFrame) -> dict[str, Any]:
    banner("ETAPE 3 — ORIGINE (toutes classes, normalisation typo uniquement)")
    df_sub = df[df["ORIGINE_clean"].notna() & (df["OMV"] == "OUI") & (df["Description"] != "")].copy()
    df_sub = _filter_classes(df_sub, "ORIGINE_clean", CONFIG["min_samples"])
    result = _train_and_eval(df_sub, "ORIGINE_clean", "ORIGINE", TABULAR_CONFIG["ORIGINE"])
    _print_test_metrics(result, "Résultats test — ORIGINE")
    result["label"] = "ORIGINE"
    return result


def run_pipeline_service(df: pd.DataFrame, agent_to_service: dict[str, str]) -> dict[str, Any]:
    banner("ETAPE 2 — SERVICE (lookup déterministe)")

    n_known = int(df["SERVICE_DETERMINISTE"].notna().sum())
    pct_known = n_known / max(len(df), 1) * 100

    section("Couverture du lookup")
    metric_line("Lignes totales", str(len(df)))
    metric_line("Demandeurs identifiés → service spécifique", f"{n_known} ({pct_known:.1f}%)")
    metric_line("Non identifiés → TOUS", f"{len(df) - n_known} ({100 - pct_known:.1f}%)")

    return {
        "label": "SERVICE",
        "agent_to_service": agent_to_service,
    }


# ============================================================================
# 7. SAUVEGARDE & CHARGEMENT
# ============================================================================

def save_artifact(key: str, result: dict[str, Any], agent_to_service: dict[str, str] | None = None) -> None:
    if key == "SERVICE":
        bundle = {
            "agent_to_service": agent_to_service or {},
            "config": CONFIG,
        }
    else:
        bundle = {
            "model": result["model"],
            "tfidf": result["tfidf"],
            "ohe": result["ohe"],
            "label_encoder": result["label_encoder"],
            "cat_cols": result["cat_cols"],
            "target_col": result["target_col"],
            "config": CONFIG,
            "service_map": SERVICE_MAP,
            "origine_map": ORIGINE_MAP,
            "villes": VILLES,
        }
        if agent_to_service is not None:
            bundle["agent_to_service"] = agent_to_service

    with open(ARTIFACTS[key], "wb") as handle:
        pickle.dump(bundle, handle)


def load_inference_bundle(key: str) -> dict[str, Any]:
    if key in _INFERENCE_CACHE:
        return _INFERENCE_CACHE[key]

    with open(ARTIFACTS[key], "rb") as handle:
        bundle = pickle.load(handle)

    _INFERENCE_CACHE[key] = bundle
    return bundle


# ============================================================================
# 8. INFÉRENCE
# ============================================================================

def _prepare_inference_row(
    description: str,
    demandeur: str = "INCONNU",
    cause: str = "INCONNU",
    traite: str = "INCONNU",
    urgence: Any = None,
) -> pd.DataFrame:
    demandeur_norm = normalize_demandeur_csv(demandeur) if "," in demandeur else demandeur.strip().upper()
    urgence_num = -1.0
    if urgence is not None:
        match = re.search(r"(\d)", str(urgence))
        urgence_num = float(match.group(1)) if match else -1.0

    desc_clean = re.sub(r"_x000D_", " ", description)
    desc_clean = re.sub(r"\s+", " ", desc_clean).strip()
    ville_match = re.search(VILLE_PATTERN, desc_clean.upper())
    ville = ville_match.group(1) if ville_match else "NON_PRECISEE"
    urgence_cat = str(int(urgence_num)) if urgence_num >= 0 else "INCONNU"

    return pd.DataFrame([{
        "Description": desc_clean,
        "CAUSE": cause.strip().upper() if cause else "INCONNU",
        "TRAITE": traite.strip().upper() if traite else "INCONNU",
        "DEMANDEUR": demandeur_norm or "INCONNU",
        "VILLE": ville,
        "URGENCE_NUM": urgence_num,
        "URGENCE_CAT": urgence_cat,
    }])


def _predict_from_bundle(bundle: dict[str, Any], row: pd.DataFrame, top_n: int) -> dict[str, Any]:
    x = transform_features(row, bundle["tfidf"], bundle["ohe"], bundle["cat_cols"])
    probs = bundle["model"].predict_proba(x)[0]
    top_idx = np.argsort(probs)[::-1][:top_n]
    labels = bundle["label_encoder"].inverse_transform(top_idx)
    top_preds = [{"classe": label, "probabilite": float(probs[idx])} for label, idx in zip(labels, top_idx)]
    return {
        "prediction": top_preds[0]["classe"],
        "confidence": float(top_preds[0]["probabilite"]),
        "top_n": top_preds,
        "probabilities": probs,
        "labels": bundle["label_encoder"].classes_,
    }


def _infer_single(key: str, row: pd.DataFrame, top_n: int) -> dict[str, Any]:
    return _predict_from_bundle(load_inference_bundle(key), row, top_n)


def predict(
    description: str,
    demandeur: str = "INCONNU",
    cause: str = "INCONNU",
    traite: str = "INCONNU",
    urgence: Any = None,
    top_n: int = 3,
) -> dict[str, Any]:
    row = _prepare_inference_row(description, demandeur, cause, traite, urgence)

    omv = _infer_single("OMV", row, top_n=2)

    if omv["prediction"] == "NON":
        return {
            "omv": {
                "prediction": "NON",
                "confidence": round(omv["confidence"] * 100, 1),
                "top_n": [{"classe": item["classe"], "probabilite": round(item["probabilite"], 4)} for item in omv["top_n"]],
            },
            "service": None,
            "origine": None,
        }

    # SERVICE : lookup déterministe (uniquement si OMV = OUI)
    service_bundle = load_inference_bundle("SERVICE")
    agent_to_service = service_bundle.get("agent_to_service", {})
    dem_norm = row["DEMANDEUR"].iloc[0]
    deterministic_service = lookup_service(dem_norm, agent_to_service)
    selected_service = deterministic_service if deterministic_service else "TOUS"
    service_source = "deterministe" if deterministic_service else "tous"

    origine = _infer_single("ORIGINE", row, top_n=top_n)

    return {
        "omv": {
            "prediction": "OUI",
            "confidence": round(omv["confidence"] * 100, 1),
            "top_n": [{"classe": item["classe"], "probabilite": round(item["probabilite"], 4)} for item in omv["top_n"]],
        },
        "service": {
            "prediction": selected_service,
            "confidence": 100.0,
            "source": service_source,
            "top_n": [{"classe": selected_service, "probabilite": 1.0}],
        },
        "origine": {
            "prediction": origine["prediction"],
            "confidence": round(origine["confidence"] * 100, 1),
            "top_n": [{"classe": item["classe"], "probabilite": round(item["probabilite"], 4)} for item in origine["top_n"]],
        },
    }


# ============================================================================
# 9. TUNING
# ============================================================================

def _make_tuning_classifier(clf_name: str, params: dict) -> Any:
    rs = CONFIG["random_state"]
    if clf_name == "logreg":
        return LogisticRegression(**params, class_weight="balanced", n_jobs=-1, random_state=rs)
    if clf_name == "linearsvc":
        return CalibratedClassifierCV(
            LinearSVC(**params, class_weight="balanced", random_state=rs, max_iter=3000, dual="auto")
        )
    if clf_name == "random_forest":
        return RandomForestClassifier(**params, class_weight="balanced", n_jobs=-1, random_state=rs)
    raise ValueError(f"Classifieur inconnu : {clf_name}")


def _param_combos(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    return [dict(zip(grid.keys(), combo)) for combo in itertools.product(*grid.values())]


def run_tuning(df: pd.DataFrame, target_col: str, target_key: str, cat_cols: list[str]) -> None:
    banner(f"TUNING — {target_key}")

    df_sub = df[df[target_col].notna() & (df["Description"] != "")].copy()
    df_sub = _filter_classes(df_sub, target_col, CONFIG["min_samples"])

    label_encoder = LabelEncoder()
    df_sub["target_encoded"] = label_encoder.fit_transform(df_sub[target_col])

    stratify = (
        df_sub["target_encoded"]
        if df_sub["target_encoded"].value_counts().min() >= 2
        else None
    )
    df_train, df_test = train_test_split(
        df_sub, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=stratify
    )

    tfidf, ohe = fit_feature_pipeline(df_train, cat_cols, CONFIG["tfidf"][target_key])
    x_train = transform_features(df_train, tfidf, ohe, cat_cols)
    x_test = transform_features(df_test, tfidf, ohe, cat_cols)
    y_train = df_train["target_encoded"].to_numpy(dtype=np.int64)
    y_test = df_test["target_encoded"].to_numpy(dtype=np.int64)

    summary: list[tuple[str, float, float, dict]] = []

    for clf_name, clf_config in TUNING_GRIDS.items():
        section(f"Classifieur : {clf_config['label']}")
        combos = _param_combos(clf_config["grids"].get(target_key, {}))
        print(f"  {len(combos)} combinaison(s) à tester...")

        best_f1 = -1.0
        best_acc = 0.0
        best_params: dict = {}

        for params in combos:
            model = _make_tuning_classifier(clf_name, params)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_acc = acc
                best_params = params

        metric_line("Meilleur F1 macro", f"{best_f1:.4f}")
        metric_line("Accuracy associée", f"{best_acc:.4f}")
        metric_line("Meilleurs params", str(best_params))
        summary.append((clf_config["label"], best_f1, best_acc, best_params))

    summary.sort(key=lambda x: x[1], reverse=True)
    section(f"Classement final — {target_key}")
    for rank, (label, f1, acc, params) in enumerate(summary, 1):
        print(f"  #{rank}  {label:<25}  F1={f1:.4f}  Acc={acc:.4f}  {params}")


NGRAM_EXPERIMENT_GRID = {
    "ngram_ranges": [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
    "svc_C_values": [0.1, 0.5, 1.0, 5.0],
    "max_features_values": [8000, 12000, 20000],
}


def run_ngram_experiment(df: pd.DataFrame, target_col: str, target_key: str, cat_cols: list[str]) -> None:
    banner(f"EXPÉRIMENTATION N-GRAMS × LinearSVC — {target_key}")

    df_sub = df[df[target_col].notna() & (df["Description"] != "")].copy()
    df_sub = _filter_classes(df_sub, target_col, CONFIG["min_samples"])

    label_encoder = LabelEncoder()
    df_sub["target_encoded"] = label_encoder.fit_transform(df_sub[target_col])

    stratify = (
        df_sub["target_encoded"]
        if df_sub["target_encoded"].value_counts().min() >= 2
        else None
    )
    df_train, df_test = train_test_split(
        df_sub, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"], stratify=stratify
    )
    y_train = df_train["target_encoded"].to_numpy(dtype=np.int64)
    y_test = df_test["target_encoded"].to_numpy(dtype=np.int64)

    rs = CONFIG["random_state"]
    results: list[tuple[float, float, tuple, int, float]] = []

    total = (
        len(NGRAM_EXPERIMENT_GRID["ngram_ranges"])
        * len(NGRAM_EXPERIMENT_GRID["max_features_values"])
        * len(NGRAM_EXPERIMENT_GRID["svc_C_values"])
    )
    print(f"  {total} combinaisons à tester (ngram × max_features × C)...\n")

    for ngram_range in NGRAM_EXPERIMENT_GRID["ngram_ranges"]:
        for max_features in NGRAM_EXPERIMENT_GRID["max_features_values"]:
            tfidf_cfg = {**CONFIG["tfidf"], "ngram_range": ngram_range, "max_features": max_features}
            tfidf, ohe = fit_feature_pipeline(df_train, cat_cols, tfidf_cfg)
            x_train = transform_features(df_train, tfidf, ohe, cat_cols)
            x_test = transform_features(df_test, tfidf, ohe, cat_cols)

            for C in NGRAM_EXPERIMENT_GRID["svc_C_values"]:
                model = CalibratedClassifierCV(
                    LinearSVC(C=C, class_weight="balanced", random_state=rs, max_iter=3000, dual="auto")
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
                acc = accuracy_score(y_test, y_pred)
                results.append((f1, acc, ngram_range, max_features, C))

    results.sort(key=lambda x: x[0], reverse=True)

    section("Top 10 combinaisons")
    print(f"  {'ngram':<10} {'max_feat':<10} {'C':<6} {'F1 macro':<12} {'Accuracy'}")
    print("  " + "-" * 55)
    for f1, acc, ngram, mf, C in results[:10]:
        print(f"  {str(ngram):<10} {mf:<10} {C:<6} {f1:.4f}       {acc:.4f}")

    section("Classement par ngram_range (meilleur F1)")
    best_per_ngram: dict[tuple, tuple[float, float, int, float]] = {}
    for f1, acc, ngram, mf, C in results:
        if ngram not in best_per_ngram or f1 > best_per_ngram[ngram][0]:
            best_per_ngram[ngram] = (f1, acc, mf, C)
    for ngram in NGRAM_EXPERIMENT_GRID["ngram_ranges"]:
        if ngram in best_per_ngram:
            f1, acc, mf, C = best_per_ngram[ngram]
            print(f"  {str(ngram):<10}  F1={f1:.4f}  Acc={acc:.4f}  max_features={mf}  C={C}")

    best_f1, best_acc, best_ngram, best_mf, best_C = results[0]
    section("Meilleure configuration globale")
    metric_line("ngram_range", str(best_ngram))
    metric_line("max_features", str(best_mf))
    metric_line("C (LinearSVC)", str(best_C))
    metric_line("F1 macro", f"{best_f1:.4f}")
    metric_line("Accuracy", f"{best_acc:.4f}")


def run_ngram_experiment_all(df: pd.DataFrame) -> None:
    run_ngram_experiment(df, "OMV", "OMV", TABULAR_CONFIG["OMV"])
    df_omv = df[df["OMV"] == "OUI"].copy()
    run_ngram_experiment(df_omv, "ORIGINE_clean", "ORIGINE", TABULAR_CONFIG["ORIGINE"])


def run_tuning_all(df: pd.DataFrame) -> None:
    run_tuning(df, "OMV", "OMV", TABULAR_CONFIG["OMV"])
    df_omv = df[df["OMV"] == "OUI"].copy()
    run_tuning(df_omv, "ORIGINE_clean", "ORIGINE", TABULAR_CONFIG["ORIGINE"])


# ============================================================================
# 10. RAPPORT
# ============================================================================

def _gain(current: float, baseline: float | None) -> str:
    if baseline is None:
        return "  —"
    return f"{current - baseline:+.3f}"


def print_comparison_table(results: dict[str, dict[str, Any]]) -> None:
    banner("TABLEAU COMPARATIF BASELINE vs VERSION RAPIDE")

    print("╔══════════════════╦══════════════╦══════════════╦════════════╗")
    print("║ Cible            ║ Baseline LR  ║ Version rapide║ Gain F1    ║")
    print("╠══════════════════╬══════════════╬══════════════╬════════════╣")

    rows = [
        ("OMV", "OMV", results["OMV"]["metrics"]),
        ("ORIGINE", "ORIGINE", results["ORIGINE"]["metrics"]),
    ]
    for display, baseline_key, metrics in rows:
        baseline = BASELINE_METRICS.get(baseline_key) if baseline_key else None
        f1_current = metrics.get("f1_macro", float("nan"))
        f1_baseline = baseline["f1_macro"] if baseline else None
        gain = _gain(f1_current, f1_baseline) if f1_baseline is not None else "  —"
        f1_base_str = f"{f1_baseline:.4f}" if f1_baseline is not None else "  N/A"
        print(
            f"║ {display:<16} ║   {f1_base_str:<10} ║   {f1_current:.4f}     ║  {gain:<9} ║"
        )
    print(f"║ {'SERVICE':<16} ║   {'N/A':<10} ║   lookup JSON     ║  {'  —':<9} ║")

    print("╚══════════════════╩══════════════╩══════════════╩════════════╝")


def print_manual_tests() -> None:
    banner("CAS DE TESTS MANUELS")

    for case in MANUAL_TEST_CASES:
        result = predict(
            description=case["description"],
            demandeur=case.get("demandeur", "INCONNU"),
            cause=case.get("cause", "INCONNU"),
            traite=case.get("traite", "INCONNU"),
            top_n=3,
        )

        omv_ok = "✓" if result["omv"]["prediction"] == case["expected_omv"] else "✗"

        print(f"\n[{case['label']}]")
        print(
            f"OMV     {omv_ok}  prédit={result['omv']['prediction']} "
            f"(conf {result['omv']['confidence']:.0f}%)  attendu={case['expected_omv']}"
        )

        if result["service"] is None:
            print("SERVICE —  (non applicable, OMV=NON)")
            print("ORIGINE —  (non applicable, OMV=NON)")
        else:
            svc_ok = "✓" if result["service"]["prediction"] == case["expected_service"] else "✗"
            ori_ok = "✓" if result["origine"]["prediction"] == case["expected_origine"] else "✗"
            svc_src = f"[{result['service']['source'].upper()}]"
            print(
                f"SERVICE {svc_ok}  prédit={result['service']['prediction']} {svc_src}  "
                f"attendu={case['expected_service']}"
            )
            print(
                f"ORIGINE {ori_ok}  prédit={result['origine']['prediction']}  "
                f"attendu={case['expected_origine']}"
            )
            print(
                "         top 3 : "
                + " | ".join(
                    f"{item['classe']} ({item['probabilite']:.2f})"
                    for item in result["origine"]["top_n"]
                )
            )


# ============================================================================
# 10. MAIN
# ============================================================================

def main(comment: str = "", tune: bool = False, ngram_experiment: bool = False) -> None:
    banner("INITIALISATION")
    print("  Backend modèle : TF-IDF (par cible) + OneHotEncoder + LinearSVC (calibré)")

    banner("CHARGEMENT DES DONNÉES")
    df_raw = load_all_sources()
    agent_to_service = build_agent_to_service(NOM_SERVICE_FILE)
    print(f"  Agents référencés dans le JSON : {len(agent_to_service)}")

    df = clean(df_raw, agent_to_service)
    metric_line("Lignes totales après nettoyage", str(len(df)))
    det_rate = df["SERVICE_DETERMINISTE"].notna().mean() * 100
    metric_line("Demandeurs connus via JSON (%)", f"{det_rate:.1f}%")

    if ngram_experiment:
        run_ngram_experiment_all(df)
        return

    if tune:
        run_tuning_all(df)
        return

    results: dict[str, dict[str, Any]] = {}
    results["OMV"] = run_pipeline_omv(df)
    results["SERVICE"] = run_pipeline_service(df, agent_to_service)
    results["ORIGINE"] = run_pipeline_origine(df)

    section("Sauvegarde des artefacts")
    save_artifact("OMV", results["OMV"])
    save_artifact("SERVICE", results["SERVICE"], agent_to_service=agent_to_service)
    save_artifact("ORIGINE", results["ORIGINE"])
    for key in ["OMV", "SERVICE", "ORIGINE"]:
        metric_line(key, ARTIFACTS[key])

    print_comparison_table(results)
    log_results(results, comment=comment)
    print_manual_tests()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de classification d'incidents CIRVIE.")
    
    # Metadata
    parser.add_argument("--comment", "-m", type=str, default="", help="Commentaire pour le run")
    
    # Hyperparamètres
    parser.add_argument("--test-size", type=float, help="Ratio du jeu de test (ex: 0.2)")
    parser.add_argument("--max-features", type=int, help="Nombre max de features TF-IDF (ex: 12000)")
    parser.add_argument("--c-omv", type=float, help="Régularisation C pour OMV")
    parser.add_argument("--c-origine", type=float, help="Régularisation C pour ORIGINE")

    # Tuning
    parser.add_argument("--tune", action="store_true", help="Comparer LogReg / LinearSVC / RandomForest sur une grille de paramètres")
    parser.add_argument("--ngram-experiment", action="store_true", help="Tester LinearSVC avec différentes configs n-grams TF-IDF")

    args = parser.parse_args()

    update_config_from_args(args)
    main(comment=args.comment, tune=args.tune, ngram_experiment=args.ngram_experiment)
