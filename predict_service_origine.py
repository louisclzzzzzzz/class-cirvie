"""
Prédiction multi-classes : SERVICE et ORIGINE
Pipeline : TF-IDF (Description) → Logistic Regression (C=5, balanced)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
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

# Normalisation : strip + majuscules
df['SERVICE_raw'] = df['SERVICE'].str.strip().str.upper()
df['ORIGINE_raw'] = df['ORIGINE'].str.strip().str.upper()

# Fusion des doublons typographiques
ORIGINE_MAPPING = {
    'BASE DE DONNEES'       : 'BASE DE DONNÉES',
    'ASSISTANCE UTILISATEUR': 'ASSISTANCE UTILISATEURS',
    'PARAMETRAGE'           : 'PARAMÉTRAGE',
    'ERREUR SAISIE'         : 'ERREUR DE SAISIE',
    'TELEPHONE'             : 'TÉLÉPHONE',
    'DONNEES'               : 'DONNÉES',
    'FISCALITE'             : 'FISCALITÉ',
}
SERVICE_MAPPING = {
    'DÉCÈS' : 'DECES',
}
df['SERVICE_clean'] = df['SERVICE_raw'].replace(SERVICE_MAPPING)
df['ORIGINE_clean'] = df['ORIGINE_raw'].replace(ORIGINE_MAPPING)

print(f"Dataset total : {len(df)} lignes")
print(f"  SERVICE labelisé : {df['SERVICE_clean'].notnull().sum()}")
print(f"  ORIGINE labelisé : {df['ORIGINE_clean'].notnull().sum()}")


# ─────────────────────────────────────────────
# 2. PARAMÈTRES COMMUNS
# ─────────────────────────────────────────────
MIN_SAMPLES = 30      # classes avec moins d'exemples → ignorées (trop peu de données)
TEST_SIZE   = 0.2
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# 3. FONCTION D'ENTRAÎNEMENT GÉNÉRIQUE
# ─────────────────────────────────────────────
def train_and_evaluate(df, target_col, min_samples=MIN_SAMPLES):
    """
    Entraîne un classifieur TF-IDF + LR sur une cible multi-classes.
    Retourne le modèle, le vectorizer, et les métriques.
    """
    # Filtrer lignes avec label et description non vides
    df_sub = df[df[target_col].notnull() & (df['Description'] != '')].copy()

    # Supprimer les classes sous-représentées (pas assez d'exemples pour apprendre)
    vc = df_sub[target_col].value_counts()
    valid_classes = vc[vc >= min_samples].index
    df_ignored = df_sub[~df_sub[target_col].isin(valid_classes)]
    df_sub = df_sub[df_sub[target_col].isin(valid_classes)].copy()

    print(f"\n{'─'*55}")
    print(f"Cible         : {target_col}")
    print(f"Lignes utiles : {len(df_sub)}  ({len(df_ignored)} ignorées — classes rares)")
    print(f"Nb classes    : {df_sub[target_col].nunique()}")

    X_raw = df_sub['Description']
    y     = df_sub[target_col]

    # TF-IDF : transforme le texte en vecteur de fréquences de tokens
    #   max_features=5000 : garde les 5 000 tokens les plus informatifs
    #   ngram_range=(1,2) : unigrammes + bigrammes
    #   sublinear_tf=True : log(tf) — atténue les mots très fréquents
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        stop_words=FRENCH_STOP_WORDS,
    )
    X_vec = tfidf.fit_transform(X_raw)

    # Split stratifié : garde les proportions de chaque classe
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Logistic Regression
    #   C=5    : moins de régularisation → meilleur sur ce problème
    #   class_weight='balanced' : compense les classes déséquilibrées
    model = LogisticRegression(C=5.0, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)

    # ── Évaluation ──
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average='macro')
    f1w  = f1_score(y_test, y_pred, average='weighted')

    print(f"\nRésultats test set ({len(y_test)} lignes) :")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  F1 macro    : {f1m:.4f}   (moyenne non-pondérée par classe)")
    print(f"  F1 weighted : {f1w:.4f}   (moyenne pondérée par support)")

    print("\nDétail par classe (triées par volume) :")
    report = classification_report(y_test, y_pred, output_dict=True)
    rows = [
        (cls, v['precision'], v['recall'], v['f1-score'], int(v['support']))
        for cls, v in report.items()
        if cls not in ['accuracy', 'macro avg', 'weighted avg']
    ]
    rows_sorted = sorted(rows, key=lambda x: -x[4])
    print(f"  {'Classe':<35} {'Prec':>5} {'Rec':>5} {'F1':>5} {'N':>5}")
    for cls, prec, rec, f1, sup in rows_sorted:
        flag = " ⚠" if f1 < 0.50 else ""
        print(f"  {cls:<35} {prec:>5.2f} {rec:>5.2f} {f1:>5.2f} {sup:>5}{flag}")

    # ── Confusions principales ──
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("\nConfusions > 5 fois :")
    found = False
    for i, ci in enumerate(model.classes_):
        for j, cj in enumerate(model.classes_):
            if i != j and cm[i, j] > 5:
                print(f"  {ci:30s} → {cj:30s} : {cm[i,j]}×")
                found = True
    if not found:
        print("  Aucune confusion majeure.")

    metrics = {
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w
    }

    return tfidf, model, valid_classes, metrics


# ─────────────────────────────────────────────
# 4. ENTRAÎNEMENT
# ─────────────────────────────────────────────
tfidf_svc, model_svc, classes_svc, metrics_svc = train_and_evaluate(df, 'SERVICE_clean')
tfidf_ori, model_ori, classes_ori, metrics_ori = train_and_evaluate(df, 'ORIGINE_clean')

# Enregistrement des métriques
log_results({"metrics": metrics_svc, "label": "SERVICE"}, pipeline_name="predict_service_origine.py")
log_results({"metrics": metrics_ori, "label": "ORIGINE"}, pipeline_name="predict_service_origine.py")


# ─────────────────────────────────────────────
# 5. CAS DE TESTS MANUELS
# ─────────────────────────────────────────────
TEST_CASES = [
    {
        "label"      : "Rachat contrat individuel Paris",
        "description": "Rachat partiel sur le contrat 30421. Situation attendue : courrier envoyé au sociétaire. "
                       "Situation obtenue : absence de courrier de rachat. Sociétaire domicilié à Paris.",
        "expected_service": "SER IND PARIS",
        "expected_origine" : "BASE DE DONNÉES",
    },
    {
        "label"      : "Connexion / mot de passe",
        "description": "Blocage connexion CIRVIE PF5. Mot de passe incorrect depuis ce matin. "
                       "L'utilisateur n'arrive plus à se connecter à l'application.",
        "expected_service": "INCONNU",
        "expected_origine" : "ASSISTANCE UTILISATEURS",
    },
    {
        "label"      : "Décès sociétaire",
        "description": "Suite au décès du sociétaire, la mise à jour du dossier bénéficiaire n'a pas été effectuée. "
                       "Le capital décès n'apparaît pas dans CIRVIE.",
        "expected_service": "DECES",
        "expected_origine" : "BASE DE DONNÉES",
    },
    {
        "label"      : "Job batch en échec",
        "description": "Traitement batch nocturne en échec depuis hier soir. "
                       "Erreur de programme lors du calcul des provisions. Relance nécessaire.",
        "expected_service": "INCONNU",
        "expected_origine" : "PROGRAMME",
    },
    {
        "label"      : "Sinistre assurance collective Nancy",
        "description": "Sinistre déclaré par un sociétaire de Nancy. Situation obtenue : "
                       "le sinistre n'est pas visible dans CIRVIE malgré la saisie correcte. "
                       "Contrat collectif, référence 78421.",
        "expected_service": "SER IND NANCY",
        "expected_origine" : "ERREUR DE SAISIE",
    },
    {
        "label"      : "Erreur de saisie rente",
        "description": "Rente viagère saisie avec un montant incorrect. "
                       "L'utilisateur a commis une erreur de saisie lors de la mise à jour du contrat. "
                       "Correction nécessaire par forçage.",
        "expected_service": "RENTES",
        "expected_origine" : "ERREUR DE SAISIE",
    },
]

print(f"\n\n{'='*60}")
print("CAS DE TESTS MANUELS")
print('='*60)

for tc in TEST_CASES:
    x = tc['description']

    # Prédiction SERVICE
    x_svc      = tfidf_svc.transform([x])
    probas_svc = model_svc.predict_proba(x_svc)[0]
    top3_svc   = sorted(zip(model_svc.classes_, probas_svc), key=lambda t: -t[1])[:3]
    pred_svc   = top3_svc[0][0]

    # Prédiction ORIGINE
    x_ori      = tfidf_ori.transform([x])
    probas_ori = model_ori.predict_proba(x_ori)[0]
    top3_ori   = sorted(zip(model_ori.classes_, probas_ori), key=lambda t: -t[1])[:3]
    pred_ori   = top3_ori[0][0]

    print(f"\n  [{tc['label']}]")
    print(f"  Texte : {x[:100]}...")

    ok_svc = "✓" if pred_svc == tc['expected_service'] else "✗"
    ok_ori = "✓" if pred_ori == tc['expected_origine'] else "✗"

    print(f"\n  SERVICE  {ok_svc} | Attendu: {tc['expected_service']:<25} | Prédit: {pred_svc}")
    print(f"    Top 3 : " + " | ".join(f"{c} ({p:.2f})" for c, p in top3_svc))

    print(f"\n  ORIGINE  {ok_ori} | Attendu: {tc['expected_origine']:<25} | Prédit: {pred_ori}")
    print(f"    Top 3 : " + " | ".join(f"{c} ({p:.2f})" for c, p in top3_ori))


# ─────────────────────────────────────────────
# 6. SAUVEGARDE
# ─────────────────────────────────────────────
pipeline_svc = {"tfidf": tfidf_svc, "model": model_svc, "valid_classes": list(classes_svc)}
pipeline_ori = {"tfidf": tfidf_ori, "model": model_ori, "valid_classes": list(classes_ori)}

with open("model_service.pkl", "wb") as f:
    pickle.dump(pipeline_svc, f)
with open("model_origine.pkl", "wb") as f:
    pickle.dump(pipeline_ori, f)

print("\n\nModèles sauvegardés : model_service.pkl | model_origine.pkl")


# ─────────────────────────────────────────────
# 7. FONCTION RÉUTILISABLE
# ─────────────────────────────────────────────
def predict_all(description: str, top_n: int = 3) -> dict:
    """
    Prédit SERVICE et ORIGINE à partir d'une description libre.
    Retourne les top_n classes les plus probables pour chaque cible.

    Paramètres
    ----------
    description : str   Texte de l'incident
    top_n       : int   Nombre de suggestions à retourner (défaut : 3)

    Retourne
    --------
    dict avec 'service' et 'origine', chacun contenant une liste de
    {'classe': str, 'probabilite': float}
    """
    if _PF5_RE.search(description) or _PWD_RE.search(description):
        return {"service": [], "origine": []}

    results = {}
    for name, tfidf, model in [
        ('service', tfidf_svc, model_svc),
        ('origine', tfidf_ori, model_ori),
    ]:
        x_vec  = tfidf.transform([description])
        probas = model.predict_proba(x_vec)[0]
        top    = sorted(zip(model.classes_, probas), key=lambda t: -t[1])[:top_n]
        results[name] = [
            {"classe": c, "probabilite": round(float(p), 4)}
            for c, p in top
        ]
    return results


# Démonstration
if __name__ == "__main__":
    res = predict_all(
        "Rachat partiel du contrat, situation obtenue incorrecte, le courrier n'a pas été généré.",
        top_n=3
    )
    print("\nExemple predict_all() :")
    for cible, suggestions in res.items():
        print(f"  {cible.upper()} :")
        for s in suggestions:
            print(f"    {s['classe']:<35} {s['probabilite']:.4f}")
