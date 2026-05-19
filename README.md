# Inc-classifier

Projet de classification automatique d'incidents CIRVIE.

L'objectif est de prédire, à partir du texte d'un ticket et de quelques métadonnées, trois informations métier :

- `INTERVENTION OMV` : classification binaire `OUI / NON`
- `SERVICE` : service cible de traitement (`TOUS` ou service spécifique du demandeur)
- `ORIGINE` : origine fonctionnelle ou technique de l'incident

## Script principal : `main.py`

### Entraînement

```bash
python3 main.py
```

Options disponibles :

| Option | Description | Exemple |
|--------|-------------|---------|
| `--comment` / `-m` | Commentaire associé au run | `-m "test C=2"` |
| `--test-size` | Ratio du jeu de test | `--test-size 0.2` |
| `--max-features` | Nombre max de features TF-IDF | `--max-features 12000` |
| `--c-omv` | Régularisation C pour le modèle OMV | `--c-omv 2.0` |
| `--c-service` | Régularisation C pour le modèle SERVICE | `--c-service 5.0` |
| `--c-origine` | Régularisation C pour le modèle ORIGINE | `--c-origine 5.0` |

Le script :

1. Charge `data3.csv` et `nom_service.json`
2. Entraîne les trois modèles (OMV, SERVICE, ORIGINE)
3. Affiche les métriques et un rapport de classification
4. Exécute des cas de tests manuels
5. Sauvegarde les artefacts

Artefacts générés :

- `model_fast_omv.pkl`
- `model_fast_service.pkl`
- `model_fast_origine.pkl`

### Inférence

Après entraînement, importer et appeler `predict(...)` :

```python
from main import predict

result = predict(
    description="Rachat partiel du contrat. Le courrier n'a pas été généré.",
    demandeur="TASSINE-BELLOUT, Anne",
    cause="BASE DE DONNÉES",
    traite="MOA OMVIE",
    urgence="2",
    top_n=3,
)

print(result)
```

Structure retournée :

```python
{
    "omv": {
        "prediction": "OUI",
        "confidence": 98.4,        # pourcentage
        "top_n": [{"classe": "OUI", "probabilite": 0.984}, ...]
    },
    "service": {
        "prediction": "SER IND PARIS",
        "confidence": 91.2,
        "source": "deterministe",  # "deterministe" ou "tous"
        "top_n": [{"classe": "SPECIFIQUE", "probabilite": 0.912}, ...]
    },
    "origine": {
        "prediction": "BASE DE DONNÉES",
        "confidence": 87.1,
        "top_n": [{"classe": "BASE DE DONNÉES", "probabilite": 0.871}, ...]
    }
}
```

## Logique SERVICE

SERVICE est prédit en deux étapes :

1. **ML binaire** : le modèle décide entre `SPECIFIQUE` (incident ciblant un service précis) ou `TOUS` (incident transverse).
2. **Lookup JSON** : si le ML dit `SPECIFIQUE` et que le demandeur est référencé dans `nom_service.json`, on retourne son service associé. Sinon, `TOUS`.

| ML prédit | Demandeur dans JSON | Résultat |
|-----------|---------------------|----------|
| SPECIFIQUE | oui | service du demandeur (ex. `SER IND PARIS`) |
| SPECIFIQUE | non | `TOUS` |
| TOUS | peu importe | `TOUS` |

Les services valides sont uniquement ceux définis dans [`nom_service.json`](nom_service.json).

## Pipeline général

Chaque modèle combine :

- **TF-IDF** sur la `Description` libre
- **OneHotEncoder** sur les variables structurées : `Cause réelle`, `Traité par`, `Demandeur`, `Urgence`, ville extraite

## Données attendues

Le script lit `data3.csv` avec au minimum les colonnes :

- `Description`
- `INTERVENTION OMV`
- `SERVICE`
- `ORIGINE`
- `Traité par`
- `Cause réelle`
- `Urgence`
- `Demandeur`

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
bash install.sh
```

Ou manuellement :

```bash
python3 -m pip install -r requirements.txt
```

## Autres scripts

| Script | Rôle |
|--------|------|
| `get_data.py` | Consolide les fichiers bruts Excel/CSV en `data3.csv` |
| `replace.py` | Correction ponctuelle de labels dans `data3.csv` |
| `metrics_logger.py` | Enregistrement des métriques dans `performance_runs.csv` |
| `predict_omv.py` | Baseline binaire OMV (historique) |
| `predict_service_origine.py` | Baseline multi-classes SERVICE/ORIGINE (historique) |
| `predict_cascade.py` | Pipeline en cascade historique |

## Suivi des performances

Chaque run enregistre automatiquement ses métriques dans [`performance_runs.csv`](performance_runs.csv) via `metrics_logger.py`.

## Reconstruction du dataset

```bash
python3 get_data.py
```

Détecte les fichiers `.csv` / `.xlsx` dans `data_brut/`, harmonise les colonnes et génère `incidents_consolides.csv`.
