"""
Prédiction INTERVENTION OMV (OUI/NON)
Pipeline : TF-IDF (Description) + OneHotEncoder (SERVICE, ORIGINE) → Logistic Regression
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)
import scipy.sparse as sp
import pickle
import warnings
from metrics_logger import log_results
warnings.filterwarnings('ignore')

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

# ─────────────────────────────────────────────
# 1. CHARGEMENT & NETTOYAGE
# ─────────────────────────────────────────────
input_file = Path('data3.csv')
if input_file.exists():
    df = pd.read_csv(input_file, encoding='utf-8-sig')
else:
    df = pd.read_excel('data_inc_cirvie.xlsx')

df['Description'] = df['Description'].fillna('')
# Adaptation des noms de colonnes si besoin
if 'SERVICE' not in df.columns and 'SERVICE_clean' in df.columns:
    df['SERVICE'] = df['SERVICE_clean']
if 'ORIGINE' not in df.columns and 'ORIGINE_clean' in df.columns:
    df['ORIGINE'] = df['ORIGINE_clean']
if 'INTERVENTION OMV' not in df.columns and 'OMV' in df.columns:
    df['INTERVENTION OMV'] = df['OMV']

df['SERVICE']     = df['SERVICE'].fillna('INCONNU').str.strip().str.upper()
df['ORIGINE']     = df['ORIGINE'].fillna('INCONNU').str.strip().str.upper()

# Cible binaire : OUI → 1, NON → 0
y = (df['INTERVENTION OMV'] == 'OUI').astype(int)

print(f"Dataset : {len(df)} lignes")
print(f"Classes : OUI={y.sum()} ({y.mean():.1%})  NON={len(y)-y.sum()} ({1-y.mean():.1%})")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────

# 2a. TF-IDF sur la Description (texte libre)
#   - max_features=5000 : garde les 5000 tokens les plus informatifs
#   - ngram_range=(1,2)  : unigrammes ET bigrammes ("mot de passe" > "mot" + "de" + "passe")
#   - sublinear_tf=True  : log(tf) au lieu de tf brut, atténue les mots très fréquents
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    stop_words=FRENCH_STOP_WORDS,
)
X_text = tfidf.fit_transform(df['Description'])  # sparse matrix (n, 5000)

# 2b. One-Hot Encoding sur SERVICE et ORIGINE
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_cat = ohe.fit_transform(df[['SERVICE', 'ORIGINE']])  # sparse matrix (n, nb_cat)

# 2c. Concaténation horizontale des deux matrices sparse
X = sp.hstack([X_text, X_cat])
print(f"Shape features : {X.shape}  ({X_text.shape[1]} TF-IDF + {X_cat.shape[1]} catégorielles)")


# ─────────────────────────────────────────────
# 3. SPLIT TRAIN / TEST
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # garde les proportions OUI/NON dans chaque split
)
print(f"\nTrain : {X_train.shape[0]} lignes | Test : {X_test.shape[0]} lignes")


# ─────────────────────────────────────────────
# 4. MODÈLE : RÉGRESSION LOGISTIQUE
# ─────────────────────────────────────────────
#   - C=1.0         : force de régularisation L2 (inverse de λ, plus grand = moins régularisé)
#   - class_weight  : 'balanced' compense le déséquilibre OUI/NON
#   - max_iter=1000 : assure la convergence sur des features sparses nombreuses
model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)


# ─────────────────────────────────────────────
# 5. ÉVALUATION
# ─────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]   # probabilité d'être OUI

# --- 5a. Cross-validation (5 folds, métrique AUC-ROC) ---
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
print(f"\n=== Cross-Validation (5-fold) ===")
print(f"AUC-ROC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- 5b. Métriques sur le test set ---
auc = roc_auc_score(y_test, y_proba)
cm  = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n=== Test Set ===")
print(f"AUC-ROC  : {auc:.4f}")
print(f"Accuracy : {(tp+tn)/(tp+tn+fp+fn):.4f}")
print(f"Matrice de confusion :")
print(f"           Prédit NON  Prédit OUI")
print(f"  Réel NON    {tn:5d}      {fp:5d}   ← Faux Positifs")
print(f"  Réel OUI    {fn:5d}      {tp:5d}   ← Faux Négatifs")
print(f"\nRapport détaillé :")
print(classification_report(y_test, y_pred, target_names=['NON', 'OUI']))

# --- 5c. Cas de tests manuels ---
print("\n=== CAS DE TESTS MANUELS ===")

TEST_CASES = [
    {
        "label": "Incident métier clair (attendu: OUI)",
        "description": "Rachat partiel du contrat 30359. Situation attendue : courrier de rachat envoyé au sociétaire. Situation obtenue : absence de courrier. Date de survenance : 05/03/2023.",
        "service": "SERI",
        "origine": "BASE DE DONNÉES"
    },
    {
        "label": "Incident technique/accès (attendu: NON)",
        "description": "Blocage connexion CIRVIE PF5. Mot de passe incorrect, impossible de se connecter. Besoin de réinitialisation du mot de passe utilisateur.",
        "service": "INCONNU",
        "origine": "ASSISTANCE UTILISATEURS"
    },
    {
        "label": "Incident ambigu",
        "description": "Erreur lors de la saisie d'un sinistre pour un assuré. Le système affiche un message d'erreur PF5.",
        "service": "SERI",
        "origine": "ERREUR DE SAISIE"
    },
    {
        "label": "Incident versement/contrat (attendu: OUI)",
        "description": "Situation attendue : versement programmé enregistré. Situation obtenue : le versement n'apparaît pas dans CIRVIE malgré la saisie correcte. Référence contrat : 45821.",
        "service": "SER IND PARIS",
        "origine": "PROGRAMME"
    },
    {
        "label": "Incident paramétrage technique (attendu: NON)",
        "description": "Job batch en échec depuis ce matin. Erreur de traitement nocturne, connexion à la base interrompue. Relance nécessaire.",
        "service": "INCONNU",
        "origine": "PROGRAMME"
    },
]

for tc in TEST_CASES:
    # Transformation des features textuelles
    x_text = tfidf.transform([tc["description"]])
    x_cat  = ohe.transform([[tc["service"], tc["origine"]]])
    x      = sp.hstack([x_text, x_cat])

    proba     = model.predict_proba(x)[0, 1]
    pred      = "OUI" if proba >= 0.5 else "NON"
    confiance = max(proba, 1 - proba) * 100

    print(f"\n  [{tc['label']}]")
    print(f"  Prédiction : {pred}  (confiance {confiance:.1f}%)  | P(OUI)={proba:.3f}")


# ─────────────────────────────────────────────
# 6. FEATURES LES PLUS IMPORTANTES
# ─────────────────────────────────────────────
feat_names = tfidf.get_feature_names_out().tolist()
coefs      = model.coef_[0][:5000]

top_pos = np.argsort(coefs)[-12:][::-1]
top_neg = np.argsort(coefs)[:12]

print("\n=== FEATURES LES PLUS PRÉDICTIVES ===")
print("→ OUI :")
for i in top_pos:
    print(f"   {feat_names[i]:30s}  {coefs[i]:+.3f}")
print("→ NON :")
for i in top_neg:
    print(f"   {feat_names[i]:30s}  {coefs[i]:+.3f}")


# ─────────────────────────────────────────────
# 7. SAUVEGARDE DU MODÈLE
# ─────────────────────────────────────────────
pipeline = {"tfidf": tfidf, "ohe": ohe, "model": model}
with open("model_omv.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("\nModèle sauvegardé dans model_omv.pkl")


# ─────────────────────────────────────────────
# 8. FONCTION DE PRÉDICTION RÉUTILISABLE
# ─────────────────────────────────────────────
def predict_omv(description: str, service: str = "INCONNU", origine: str = "INCONNU") -> dict:
    """
    Prédit si un incident nécessite une intervention OMV.

    Paramètres
    ----------
    description : str   Texte libre de l'incident
    service     : str   Code service (ex: 'SERI', 'SER IND PARIS')
    origine     : str   Origine de l'incident (ex: 'TÉLÉPHONE', 'PROGRAMME')

    Retourne
    --------
    dict avec 'prediction' (OUI/NON), 'proba_oui', 'confiance'
    """
    if _PF5_RE.search(description) or _PWD_RE.search(description):
        return {"prediction": "NON", "proba_oui": 0.0, "confiance": 100.0}

    service = service.strip().upper()
    origine = origine.strip().upper()
    x_text  = tfidf.transform([description])
    x_cat   = ohe.transform([[service, origine]])
    x       = sp.hstack([x_text, x_cat])
    proba   = model.predict_proba(x)[0, 1]
    pred    = "OUI" if proba >= 0.5 else "NON"
    return {
        "prediction" : pred,
        "proba_oui"  : round(float(proba), 4),
        "confiance"  : round(float(max(proba, 1 - proba)) * 100, 1)
    }


# Exemple d'utilisation
if __name__ == "__main__":
    result = predict_omv(
        description="Contrat rachat partiel, situation obtenue incorrecte, le courrier n'a pas été généré.",
        service="SERI",
        origine="BASE DE DONNÉES"
    )
    print(f"\nExemple predict_omv() : {result}")

    # Enregistrement des métriques
    log_results(
        {
            "metrics": {
                "accuracy": (tp+tn)/(tp+tn+fp+fn),
                "auc_roc": auc
            },
            "label": "OMV"
        },
        pipeline_name="predict_omv.py",
        model_type="TF-IDF + LogReg"
    )
