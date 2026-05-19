"""
Pipeline de classification en cascade
  Étape 1 : Prédiction INTERVENTION OMV  (OUI/NON)
  Étape 2 : Prédiction SERVICE           (utilise la prédiction OMV comme feature)
  Étape 3 : Prédiction ORIGINE           (utilise la prédiction OMV comme feature)

Features utilisées :
  - Description      (TF-IDF texte libre)
  - SERVICE/ORIGINE  (OHE, selon disponibilité)
  - INTERVENTION OMV (prédite à l'étape 1 — cascade)
  - Cause réelle     (OHE)
  - Traité par       (OHE)
"""

import re
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
import warnings
from pathlib import Path
from metrics_logger import log_results
warnings.filterwarnings("ignore")

# ── Règles métier déterministes ──────────────────────────────────────────────
# PF5 = écran de connexion CIRVIE → pas d'intervention OMV, service/origine vides
# Mot de passe → réinitialisation compte, pas d'intervention OMV
_PF5_RE = re.compile(r'\bPF5\b', re.IGNORECASE)
_PWD_RE = re.compile(r'\b(mot\s+de\s+passe|password|mdp)\b', re.IGNORECASE)

FRENCH_STOP_WORDS = frozenset([
    "a", "au", "aux", "avec", "car", "ce", "ceci", "cela", "ces", "cet",
    "cette", "dans", "de", "des", "donc", "du", "elle", "elles", "en",
    "entre", "et", "eux", "ici", "il", "ils", "je", "la", "là", "le",
    "les", "leur", "leurs", "lui", "lors", "ma", "mais", "me", "même",
    "mes", "mon", "ni", "nos", "notre", "nous", "on", "or", "ou", "où",
    "par", "pas", "plus", "pour", "puis", "qu", "que", "quand", "quel",
    "quelle", "quels", "quelles", "qui", "sa", "sans", "sauf", "se",
    "ses", "si", "son", "sur", "ta", "te", "tes", "ton", "tout", "tous",
    "toute", "toutes", "très", "trop", "tu", "un", "une", "via", "voir",
    "vos", "votre", "vous", "y", "à", "été", "être", "est", "sont",
])

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing     import OneHotEncoder
from sklearn.linear_model      import LogisticRegression
from sklearn.metrics           import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

FILES = [
    "data2.csv",          # historique 2018-2024
]

# Colonnes cibles
TARGET_OMV     = "INTERVENTION OMV"
TARGET_SERVICE = "SERVICE"
TARGET_ORIGINE = "ORIGINE"

# Seuil minimum d'occurrences pour conserver une classe
MIN_SAMPLES = 30

# Ratio test
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# Mappings pour normaliser les doublons typographiques
ORIGINE_MAP = {
    "BASE DE DONNEES"       : "BASE DE DONNÉES",
    "ASSISTANCE UTILISATEUR": "ASSISTANCE UTILISATEURS",
    "PARAMETRAGE"           : "PARAMÉTRAGE",
    "ERREUR SAISIE"         : "ERREUR DE SAISIE",
    "TELEPHONE"             : "TÉLÉPHONE",
    "DONNEES"               : "DONNÉES",
    "FISCALITE"             : "FISCALITÉ",
}
SERVICE_MAP = {"DÉCÈS": "DECES"}


# ══════════════════════════════════════════════════════════════════════
# AFFICHAGE
# ══════════════════════════════════════════════════════════════════════

W = 65  # largeur de la console

def banner(title: str):
    print("\n" + "═" * W)
    print(f"  {title}")
    print("═" * W)

def section(title: str):
    print(f"\n  ── {title}")

def row(label: str, value: str, width: int = 30):
    print(f"    {label:<{width}} {value}")

def table_header(*cols, widths):
    line = "    " + "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(line)
    print("    " + "─" * (sum(widths) - 4))

def table_row(*vals, widths):
    print("    " + "".join(f"{str(v):<{w}}" for v, w in zip(vals, widths)))


# ══════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & NETTOYAGE
# ══════════════════════════════════════════════════════════════════════

def load_data(files: list) -> pd.DataFrame:
    """
    Charge et concatène plusieurs fichiers CSV ou Excel.
    Gère le fichier 2026 qui a des colonnes supplémentaires
    (Cause réelle, Traité par, Urgence, Résolution immédiate…).
    """
    frames = []
    for path in files:
        try:
            path_obj = Path(path)
            if path_obj.suffix.lower() == ".csv":
                raw = pd.read_csv(path_obj, dtype=str, encoding="utf-8-sig")
            else:
                xl = pd.ExcelFile(path_obj)
                if "TOUS LES INCIDENTS" in xl.sheet_names:
                    raw = pd.read_excel(path_obj, sheet_name="TOUS LES INCIDENTS", header=None)
                    raw.columns = [str(c) for c in raw.iloc[0]]
                    raw = raw.iloc[1:].reset_index(drop=True)
                else:
                    raw = pd.read_excel(path_obj)
            frames.append(raw)
            print(f"  Chargé : {path:<35}  ({len(raw)} lignes)")
        except FileNotFoundError:
            print(f"  ⚠ Fichier introuvable : {path}  — ignoré")

    df = pd.concat(frames, ignore_index=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisation des colonnes."""

    def norm(series):
        return series.astype(str).str.strip().str.upper().replace({"NAN": np.nan})

    df["Description"]   = df["Description"].fillna("")
    df["OMV"]           = norm(df[TARGET_OMV])
    df["SERVICE_clean"] = norm(df[TARGET_SERVICE]).replace(SERVICE_MAP)
    df["ORIGINE_clean"] = norm(df[TARGET_ORIGINE]).replace(ORIGINE_MAP)

    # Colonnes supplémentaires (présentes uniquement dans le fichier 2026)
    df["CAUSE"]    = norm(df["Cause réelle"]) if "Cause réelle" in df.columns else np.nan
    df["TRAITE"]   = norm(df["Traité par"])   if "Traité par"   in df.columns else np.nan

    # Nettoyage des valeurs invalides
    df.loc[~df["OMV"].isin(["OUI", "NON"]), "OMV"] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════
# 2. CONSTRUCTION DES FEATURES
# ══════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame,
                   tfidf: TfidfVectorizer,
                   cat_cols: list,
                   ohe: OneHotEncoder,
                   omv_col: str = None) -> sp.csr_matrix:
    """
    Assemble la matrice de features sparse :
      [TF-IDF] + [OHE catégorielles] + [OMV encodé si fourni]
    """
    X_text = tfidf.transform(df["Description"])

    cat_data = df[cat_cols].fillna("INCONNU").astype(str)
    X_cat    = ohe.transform(cat_data)

    parts = [X_text, X_cat]

    if omv_col is not None:
        # OMV binaire encodé en 0/1, reshape en colonne sparse
        omv_vec = (df[omv_col] == "OUI").astype(float).values.reshape(-1, 1)
        parts.append(sp.csr_matrix(omv_vec))

    return sp.hstack(parts, format="csr")


# ══════════════════════════════════════════════════════════════════════
# 3. ÉTAPE 1 — PRÉDICTION OMV
# ══════════════════════════════════════════════════════════════════════

def train_omv(df: pd.DataFrame):
    banner("ÉTAPE 1 — INTERVENTION OMV  (OUI / NON)")

    df_sub = df[df["OMV"].notnull() & (df["Description"] != "")].copy()
    y = (df_sub["OMV"] == "OUI").astype(int)

    section("Données")
    row("Lignes disponibles", f"{len(df_sub)}")
    row("OUI", f"{y.sum()} ({y.mean():.1%})")
    row("NON", f"{len(y)-y.sum()} ({1-y.mean():.1%})")

    # Features : texte + colonnes catégorielles disponibles
    cat_cols = ["CAUSE", "TRAITE"]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            min_df=3, sublinear_tf=True,
                            stop_words=FRENCH_STOP_WORDS)
    tfidf.fit(df_sub["Description"])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe.fit(df_sub[cat_cols].fillna("INCONNU").astype(str))

    X = build_features(df_sub, tfidf, cat_cols, ohe, omv_col=None)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_sub.index,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cv = cross_val_score(model, X_train, y_train, cv=5,
                         scoring="roc_auc", n_jobs=-1)

    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    section("Performances")
    row("AUC-ROC test",          f"{roc_auc_score(y_test, y_proba):.4f}")
    row("AUC-ROC cross-val (5×)", f"{cv.mean():.4f} ± {cv.std():.4f}")
    row("Accuracy",              f"{accuracy_score(y_test, y_pred):.4f}")
    row("Vrais positifs (TP)",   f"{tp}")
    row("Vrais négatifs (TN)",   f"{tn}")
    row("Faux positifs (FP)",    f"{fp}  (prédit OUI → réel NON)")
    row("Faux négatifs (FN)",    f"{fn}  (prédit NON → réel OUI)")

    section("Rapport par classe")
    report = classification_report(y_test, y_pred,
                                   target_names=["NON", "OUI"])
    for line in report.strip().split("\n"):
        print(f"    {line}")

    # Générer les prédictions sur TOUT le dataset (pour la cascade)
    X_all  = build_features(df_sub, tfidf, cat_cols, ohe, omv_col=None)
    omv_pred_all = pd.Series(
        model.predict(X_all), index=df_sub.index, name="OMV_PRED"
    )
    omv_pred_all = omv_pred_all.map({1: "OUI", 0: "NON"})

    return tfidf, ohe, model, cat_cols, omv_pred_all, idx_test, y_test, y_pred


# ══════════════════════════════════════════════════════════════════════
# 4. ÉTAPE 2 & 3 — SERVICE et ORIGINE (avec OMV prédit en cascade)
# ══════════════════════════════════════════════════════════════════════

def train_multiclass(df: pd.DataFrame,
                     target_col: str,
                     omv_pred: pd.Series,
                     step_label: str):
    banner(f"{step_label} — {target_col.upper()}")

    # Joindre la prédiction OMV
    df = df.copy()
    df["OMV_PRED"] = omv_pred

    df_sub = df[df[target_col].notnull() & (df["Description"] != "")].copy()

    # Supprimer les classes rares
    vc = df_sub[target_col].value_counts()
    valid = vc[vc >= MIN_SAMPLES].index
    ignored = df_sub[~df_sub[target_col].isin(valid)][target_col].value_counts()
    df_sub  = df_sub[df_sub[target_col].isin(valid)].copy()
    y       = df_sub[target_col]

    section("Données")
    row("Lignes disponibles",   f"{len(df_sub)}")
    row("Nb classes (≥30 obs)", f"{y.nunique()}")
    if len(ignored):
        row("Classes ignorées (<30)", ", ".join(ignored.index.tolist()))

    # Features : texte + catégorielles + OMV_PRED (cascade)
    cat_cols = ["CAUSE", "TRAITE", "OMV_PRED"]

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                            min_df=2, sublinear_tf=True,
                            stop_words=FRENCH_STOP_WORDS)
    tfidf.fit(df_sub["Description"])

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe.fit(df_sub[cat_cols].fillna("INCONNU").astype(str))

    # ── Sans cascade (baseline) ──────────────────────────────────────
    # Pour mesurer le gain apporté par OMV_PRED, on entraîne aussi
    # un modèle sans cette feature.
    cat_cols_base = ["CAUSE", "TRAITE"]
    ohe_base = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    ohe_base.fit(df_sub[cat_cols_base].fillna("INCONNU").astype(str))

    X_base = build_features(df_sub, tfidf, cat_cols_base, ohe_base, omv_col=None)
    X_casc = build_features(df_sub, tfidf, cat_cols,      ohe,      omv_col=None)

    X_tr_b, X_te_b, y_train, y_test = train_test_split(
        X_base, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr_c, X_te_c, _, _ = train_test_split(
        X_casc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model_base = LogisticRegression(C=5.0, class_weight="balanced", max_iter=1000)
    model_casc = LogisticRegression(C=5.0, class_weight="balanced", max_iter=1000)
    model_base.fit(X_tr_b, y_train)
    model_casc.fit(X_tr_c, y_train)

    y_pred_base = model_base.predict(X_te_b)
    y_pred_casc = model_casc.predict(X_te_c)

    acc_b  = accuracy_score(y_test, y_pred_base)
    acc_c  = accuracy_score(y_test, y_pred_casc)
    f1m_b  = f1_score(y_test, y_pred_base, average="macro")
    f1m_c  = f1_score(y_test, y_pred_casc, average="macro")
    f1w_b  = f1_score(y_test, y_pred_base, average="weighted")
    f1w_c  = f1_score(y_test, y_pred_casc, average="weighted")

    section("Comparaison  sans cascade  vs  avec cascade (OMV prédit)")
    widths = [28, 14, 14, 14]
    table_header("Métrique", "Sans cascade", "Avec cascade", "Gain", widths=widths)
    for label, vb, vc_ in [
        ("Accuracy",    acc_b,  acc_c),
        ("F1 macro",    f1m_b,  f1m_c),
        ("F1 weighted", f1w_b,  f1w_c),
    ]:
        gain = vc_ - vb
        sign = "+" if gain >= 0 else ""
        table_row(label, f"{vb:.4f}", f"{vc_:.4f}",
                  f"{sign}{gain:.4f}", widths=widths)

    section("Détail par classe (modèle avec cascade, trié par volume)")
    report = classification_report(y_test, y_pred_casc, output_dict=True)
    rows = [
        (cls, v["precision"], v["recall"], v["f1-score"], int(v["support"]))
        for cls, v in report.items()
        if cls not in ["accuracy", "macro avg", "weighted avg"]
    ]
    rows_sorted = sorted(rows, key=lambda x: -x[4])
    widths2 = [36, 8, 8, 8, 8]
    table_header("Classe", "Prec", "Recall", "F1", "N", widths=widths2)
    for cls, prec, rec, f1, sup in rows_sorted:
        flag = "  ⚠" if f1 < 0.50 else ""
        table_row(cls, f"{prec:.2f}", f"{rec:.2f}",
                  f"{f1:.2f}{flag}", sup, widths=widths2)

    section("Principales confusions (> 5 fois)")
    cm_mat = confusion_matrix(y_test, y_pred_casc, labels=model_casc.classes_)
    found  = False
    for i, ci in enumerate(model_casc.classes_):
        for j, cj in enumerate(model_casc.classes_):
            if i != j and cm_mat[i, j] > 5:
                print(f"    {ci:<30} → {cj:<30} {cm_mat[i,j]}×")
                found = True
    if not found:
        print("    Aucune confusion majeure.")

    return tfidf, ohe, model_casc, cat_cols


# ══════════════════════════════════════════════════════════════════════
# 5. CAS DE TESTS MANUELS
# ══════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "label"      : "Rachat contrat individuel Paris",
        "description": "Rachat partiel sur le contrat 30421. Situation attendue : courrier envoyé "
                       "au sociétaire. Situation obtenue : absence de courrier. "
                       "Sociétaire domicilié à Paris.",
        "cause"      : "BASE DE DONNÉES",
        "traite"     : "MOA OMVIE",
        "omv_attendu": "OUI",
        "svc_attendu": "SER IND PARIS",
        "ori_attendu": "BASE DE DONNÉES",
    },
    {
        "label"      : "Mot de passe oublié",
        "description": "Blocage connexion CIRVIE PF5. L'utilisateur a oublié son mot de passe "
                       "et ne peut plus se connecter à l'application depuis ce matin.",
        "cause"      : "MOT DE PASSE",
        "traite"     : "EXP SUPPORT HABILITATION",
        "omv_attendu": "NON",
        "svc_attendu": "—",
        "ori_attendu": "ASSISTANCE UTILISATEURS",
    },
    {
        "label"      : "Anomalie 990I décès",
        "description": "INCIDENT CIRVIE. Situation attendue : capitaux réglés corrects avec "
                       "abattement appliqué une seule fois. Situation obtenue : abattement "
                       "990I appliqué plusieurs fois sur les 4 contrats du sociétaire décédé.",
        "cause"      : "PROGRAMME",
        "traite"     : "MOA OMVIE",
        "omv_attendu": "OUI",
        "svc_attendu": "DECES",
        "ori_attendu": "ANOMALIE 990I",
    },
    {
        "label"      : "Job batch en échec",
        "description": "Le job A1P1CV45FA est en état JOBFAILURE sur SMASPRO1. "
                       "Traitement batch nocturne en échec, relance nécessaire.",
        "cause"      : "BATCH",
        "traite"     : "EXP OPÉRATIONS DOMAINE VCP",
        "omv_attendu": "NON",
        "svc_attendu": "—",
        "ori_attendu": "PROGRAMME",
    },
    {
        "label"      : "Sinistre collectif Nancy",
        "description": "Sinistre déclaré par un sociétaire de Nancy. Situation obtenue : "
                       "le sinistre n'apparaît pas dans CIRVIE malgré la saisie. "
                       "Contrat collectif, référence 78421.",
        "cause"      : "BASE DE DONNÉES",
        "traite"     : "ETD DOMAINE VIE",
        "omv_attendu": "OUI",
        "svc_attendu": "SER IND NANCY",
        "ori_attendu": "ERREUR DE SAISIE",
    },
]


def run_tests(tfidf_omv, ohe_omv, model_omv, cat_omv,
              tfidf_svc, ohe_svc, model_svc, cat_svc,
              tfidf_ori, ohe_ori, model_ori, cat_ori):

    banner("CAS DE TESTS MANUELS")

    for tc in TEST_CASES:
        print(f"\n  ┌─ {tc['label']}")
        print(f"  │  Texte : {tc['description'][:80]}...")

        # ── OMV ───────────────────────────────────────────────────────
        row_df = pd.DataFrame([{
            "Description": tc["description"],
            "CAUSE"      : tc["cause"].upper(),
            "TRAITE"     : tc["traite"].upper(),
            "OMV_PRED"   : "INCONNU",
        }])

        x_omv   = build_features(row_df, tfidf_omv, ["CAUSE", "TRAITE"], ohe_omv)
        p_omv   = model_omv.predict_proba(x_omv)[0]
        omv_pred = "OUI" if p_omv[1] >= 0.5 else "NON"
        omv_conf = max(p_omv) * 100
        omv_ok   = "✓" if omv_pred == tc["omv_attendu"] else "✗"

        print(f"  │")
        print(f"  │  OMV     {omv_ok}  prédit={omv_pred:<4} "
              f"(conf {omv_conf:.0f}%)  attendu={tc['omv_attendu']}")

        # ── SERVICE (cascade : on injecte omv_pred) ───────────────────
        row_df["OMV_PRED"] = omv_pred

        x_svc    = build_features(row_df, tfidf_svc, cat_svc, ohe_svc)
        p_svc    = model_svc.predict_proba(x_svc)[0]
        top3_svc = sorted(zip(model_svc.classes_, p_svc), key=lambda t: -t[1])[:3]
        svc_pred = top3_svc[0][0]
        svc_ok   = "✓" if svc_pred == tc["svc_attendu"] else "✗"

        top3_str = "  |  ".join(f"{c} ({p:.2f})" for c, p in top3_svc)
        print(f"  │  SERVICE {svc_ok}  prédit={svc_pred:<20} attendu={tc['svc_attendu']}")
        print(f"  │          top 3 : {top3_str}")

        # ── ORIGINE (cascade) ─────────────────────────────────────────
        x_ori    = build_features(row_df, tfidf_ori, cat_ori, ohe_ori)
        p_ori    = model_ori.predict_proba(x_ori)[0]
        top3_ori = sorted(zip(model_ori.classes_, p_ori), key=lambda t: -t[1])[:3]
        ori_pred = top3_ori[0][0]
        ori_ok   = "✓" if ori_pred == tc["ori_attendu"] else "✗"

        top3_str = "  |  ".join(f"{c} ({p:.2f})" for c, p in top3_ori)
        print(f"  │  ORIGINE {ori_ok}  prédit={ori_pred:<25} attendu={tc['ori_attendu']}")
        print(f"  └          top 3 : {top3_str}")


# ══════════════════════════════════════════════════════════════════════
# 6. SAUVEGARDE
# ══════════════════════════════════════════════════════════════════════

def save_models(tfidf_omv, ohe_omv, model_omv, cat_omv,
                tfidf_svc, ohe_svc, model_svc, cat_svc,
                tfidf_ori, ohe_ori, model_ori, cat_ori):
    banner("SAUVEGARDE DES MODÈLES")
    for fname, obj in [
        ("model_cascade_omv.pkl",     {"tfidf": tfidf_omv, "ohe": ohe_omv,
                                       "model": model_omv,  "cat_cols": cat_omv}),
        ("model_cascade_service.pkl", {"tfidf": tfidf_svc, "ohe": ohe_svc,
                                       "model": model_svc,  "cat_cols": cat_svc}),
        ("model_cascade_origine.pkl", {"tfidf": tfidf_ori, "ohe": ohe_ori,
                                       "model": model_ori,  "cat_cols": cat_ori}),
    ]:
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Sauvegardé : {fname}")


# ══════════════════════════════════════════════════════════════════════
# 7. FONCTION DE PRÉDICTION RÉUTILISABLE
# ══════════════════════════════════════════════════════════════════════

def predict(description: str,
            cause: str = "INCONNU",
            traite: str = "INCONNU",
            top_n: int = 3) -> dict:
    """
    Prédit les 3 classes en cascade à partir d'une description.

    Paramètres
    ----------
    description : str   Texte de l'incident
    cause       : str   Cause réelle (si connue)
    traite      : str   Équipe traitante (si connue)
    top_n       : int   Nombre de suggestions retournées par classe

    Retourne
    --------
    dict avec 'omv', 'service', 'origine'
    """
    # Règles métier : court-circuit avant tout appel ML
    if _PF5_RE.search(description) or _PWD_RE.search(description):
        return {
            "omv"    : {"prediction": "NON", "confiance": 100.0},
            "service": [],
            "origine": [],
        }

    # Charger les modèles sauvegardés
    pipelines = {}
    for key, fname in [("omv",     "model_cascade_omv.pkl"),
                       ("service", "model_cascade_service.pkl"),
                       ("origine", "model_cascade_origine.pkl")]:
        with open(fname, "rb") as f:
            pipelines[key] = pickle.load(f)

    row_df = pd.DataFrame([{
        "Description": description,
        "CAUSE"      : cause.strip().upper(),
        "TRAITE"     : traite.strip().upper(),
        "OMV_PRED"   : "INCONNU",
    }])

    # Étape 1 : OMV
    p  = pipelines["omv"]
    x  = build_features(row_df, p["tfidf"], p["cat_cols"], p["ohe"])
    pr = p["model"].predict_proba(x)[0]
    omv_label = "OUI" if pr[1] >= 0.5 else "NON"
    omv_conf  = round(float(max(pr)) * 100, 1)

    # Injecter OMV prédit pour la cascade
    row_df["OMV_PRED"] = omv_label

    results = {"omv": {"prediction": omv_label, "confiance": omv_conf}}

    # Étapes 2 & 3
    for key in ["service", "origine"]:
        p  = pipelines[key]
        x  = build_features(row_df, p["tfidf"], p["cat_cols"], p["ohe"])
        pr = p["model"].predict_proba(x)[0]
        top = sorted(zip(p["model"].classes_, pr), key=lambda t: -t[1])[:top_n]
        results[key] = [
            {"classe": c, "probabilite": round(float(prob), 4)}
            for c, prob in top
        ]

    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    banner("CHARGEMENT DES DONNÉES")
    df_raw = load_data(FILES)
    df     = clean(df_raw)
    print(f"\n  Total lignes fusionnées : {len(df)}")

    # ── Étape 1 : OMV ─────────────────────────────────────────────────
    (tfidf_omv, ohe_omv, model_omv,
     cat_omv, omv_pred_all,
     idx_test_omv, y_test_omv, y_pred_omv) = train_omv(df)

    # ── Étape 2 : SERVICE (cascade) ────────────────────────────────────
    (tfidf_svc, ohe_svc, model_svc,
     cat_svc) = train_multiclass(
        df, "SERVICE_clean", omv_pred_all,
        step_label="ÉTAPE 2"
    )

    # ── Étape 3 : ORIGINE (cascade) ────────────────────────────────────
    (tfidf_ori, ohe_ori, model_ori,
     cat_ori) = train_multiclass(
        df, "ORIGINE_clean", omv_pred_all,
        step_label="ÉTAPE 3"
    )

    # ── Tests manuels ──────────────────────────────────────────────────
    run_tests(
        tfidf_omv, ohe_omv, model_omv, cat_omv,
        tfidf_svc, ohe_svc, model_svc, cat_svc,
        tfidf_ori, ohe_ori, model_ori, cat_ori,
    )

    # ── Sauvegarde ─────────────────────────────────────────────────────
    save_models(
        tfidf_omv, ohe_omv, model_omv, cat_omv,
        tfidf_svc, ohe_svc, model_svc, cat_svc,
        tfidf_ori, ohe_ori, model_ori, cat_ori,
    )

    # ── Exemple predict() ──────────────────────────────────────────────
    banner("EXEMPLE — FONCTION predict()")
    res = predict(
        description="Rachat partiel du contrat, situation obtenue incorrecte, "
                    "le courrier de rachat n'a pas été généré.",
        cause="BASE DE DONNÉES",
        traite="MOA OMVIE",
    )
    print(f"  OMV     : {res['omv']['prediction']}  (confiance {res['omv']['confiance']}%)")
    print(f"  SERVICE : " + " | ".join(
        f"{s['classe']} ({s['probabilite']:.3f})" for s in res["service"]))
    print(f"  ORIGINE : " + " | ".join(
        f"{s['classe']} ({s['probabilite']:.3f})" for s in res["origine"]))
    print()
