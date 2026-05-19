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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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
        "max_features": 12000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "sublinear_tf": True,
        "strip_accents": "unicode",
    },
    "logreg": {
        "OMV": {"C": 2.0, "solver": "liblinear", "max_iter": 1500},
        "ORIGINE": {"C": 5.0, "solver": "saga", "max_iter": 2500},
    },
}

def update_config_from_args(args: argparse.Namespace):
    """Met à jour la configuration globale avec les arguments CLI."""
    if args.test_size:
        CONFIG["test_size"] = args.test_size
    if args.max_features:
        CONFIG["tfidf"]["max_features"] = args.max_features
    if args.c_omv:
        CONFIG["logreg"]["OMV"]["C"] = args.c_omv
    if args.c_service:
        CONFIG["logreg"]["SERVICE"]["C"] = args.c_service
    if args.c_origine:
        CONFIG["logreg"]["ORIGINE"]["C"] = args.c_origine

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
    "SERVICE": ["CAUSE", "TRAITE", "DEMANDEUR", "VILLE", "URGENCE_CAT"],
    "ORIGINE": ["CAUSE", "TRAITE", "DEMANDEUR", "VILLE", "URGENCE_CAT"],
}

ARTIFACTS = {
    "OMV": "model_fast_omv.pkl",
    "SERVICE": "model_fast_service.pkl",
    "ORIGINE": "model_fast_origine.pkl",
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
        "expected_service": "TOUS",
        "expected_origine": "ASSISTANCE UTILISATEURS",
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
        "expected_service": "TOUS",
        "expected_origine": "PROGRAMME",
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


def build_model(target_key: str) -> LogisticRegression:
    params = CONFIG["logreg"][target_key]
    return LogisticRegression(
        C=params["C"],
        solver=params["solver"],
        max_iter=params["max_iter"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=CONFIG["random_state"],
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

    tfidf, ohe = fit_feature_pipeline(df_train, cat_cols, CONFIG["tfidf"])
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
    df_sub = df[df["ORIGINE_clean"].notna() & (df["Description"] != "")].copy()
    df_sub = _filter_classes(df_sub, "ORIGINE_clean", CONFIG["min_samples"])
    result = _train_and_eval(df_sub, "ORIGINE_clean", "ORIGINE", TABULAR_CONFIG["ORIGINE"])
    _print_test_metrics(result, "Résultats test — ORIGINE")
    result["label"] = "ORIGINE"
    return result


def run_pipeline_service(df: pd.DataFrame, agent_to_service: dict[str, str]) -> dict[str, Any]:
    banner("ETAPE 2 — SERVICE (ML binaire : TOUS vs service du demandeur)")

    df_sub = df[df["SERVICE_clean"].notna() & (df["Description"] != "")].copy()

    # Cible binaire : TOUS ou SPECIFIQUE (le lookup donnera le service exact à l'inférence)
    df_sub["SERVICE_BINARY"] = df_sub["SERVICE_clean"].apply(
        lambda x: "TOUS" if x == "TOUS" else "SPECIFIQUE"
    )

    section("Distribution de la cible binaire")
    counts = df_sub["SERVICE_BINARY"].value_counts()
    for label, count in counts.items():
        metric_line(label, f"{count} ({count / len(df_sub) * 100:.1f}%)")

    result = _train_and_eval(df_sub, "SERVICE_BINARY", "SERVICE", TABULAR_CONFIG["SERVICE"])
    _print_test_metrics(result, "Résultats test — SERVICE (TOUS vs SPECIFIQUE)")

    result["label"] = "SERVICE"
    result["agent_to_service"] = agent_to_service
    return result


# ============================================================================
# 7. SAUVEGARDE & CHARGEMENT
# ============================================================================

def save_artifact(key: str, result: dict[str, Any], agent_to_service: dict[str, str] | None = None) -> None:
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
    origine = _infer_single("ORIGINE", row, top_n=top_n)

    service_bundle = load_inference_bundle("SERVICE")
    agent_to_service = service_bundle.get("agent_to_service", {})
    dem_norm = row["DEMANDEUR"].iloc[0]
    deterministic_service = lookup_service(dem_norm, agent_to_service)

    # ML décide : TOUS ou SPECIFIQUE
    service_binary = _predict_from_bundle(service_bundle, row, top_n=2)
    ml_says_specifique = service_binary["prediction"] == "SPECIFIQUE"

    if ml_says_specifique and deterministic_service:
        selected_service = deterministic_service
        service_source = "deterministe"
    else:
        selected_service = "TOUS"
        service_source = "tous"

    return {
        "omv": {
            "prediction": omv["prediction"],
            "confidence": round(omv["confidence"] * 100, 1),
            "top_n": [{"classe": item["classe"], "probabilite": round(item["probabilite"], 4)} for item in omv["top_n"]],
        },
        "service": {
            "prediction": selected_service,
            "confidence": round(service_binary["confidence"] * 100, 1),
            "source": service_source,
            "top_n": [{"classe": item["classe"], "probabilite": round(item["probabilite"], 4)} for item in service_binary["top_n"]],
        },
        "origine": {
            "prediction": origine["prediction"],
            "confidence": round(origine["confidence"] * 100, 1),
            "top_n": [{"classe": item["classe"], "probabilite": round(item["probabilite"], 4)} for item in origine["top_n"]],
        },
    }


# ============================================================================
# 9. RAPPORT
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
        ("SERVICE (TOUS/SPEC)", "SERVICE", results["SERVICE"]["metrics"]),
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
        svc_ok = "✓" if result["service"]["prediction"] == case["expected_service"] else "✗"
        ori_ok = "✓" if result["origine"]["prediction"] == case["expected_origine"] else "✗"
        svc_src = f"[{result['service']['source'].upper()}]"

        print(f"\n[{case['label']}]")
        print(
            f"OMV     {omv_ok}  prédit={result['omv']['prediction']} "
            f"(conf {result['omv']['confidence']:.0f}%)  attendu={case['expected_omv']}"
        )
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

def main(comment: str = "") -> None:
    banner("INITIALISATION")
    print("  Backend modèle : TF-IDF + OneHotEncoder + LogisticRegression")

    banner("CHARGEMENT DES DONNÉES")
    df_raw = load_all_sources()
    agent_to_service = build_agent_to_service(NOM_SERVICE_FILE)
    print(f"  Agents référencés dans le JSON : {len(agent_to_service)}")

    df = clean(df_raw, agent_to_service)
    metric_line("Lignes totales après nettoyage", str(len(df)))
    det_rate = df["SERVICE_DETERMINISTE"].notna().mean() * 100
    metric_line("Demandeurs connus via JSON (%)", f"{det_rate:.1f}%")

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
    parser.add_argument("--c-service", type=float, help="Régularisation C pour SERVICE")
    parser.add_argument("--c-origine", type=float, help="Régularisation C pour ORIGINE")
    
    args = parser.parse_args()
    
    update_config_from_args(args)
    main(comment=args.comment)
