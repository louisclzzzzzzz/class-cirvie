import pandas as pd
import os
import unicodedata
from pathlib import Path

# Dossier contenant les fichiers Excel ou CSV bruts
input_folder = "data_brut"  # Modifie ce chemin si besoin

# Colonnes cibles
TARGET_COLS = ["N° d'Incident", "INTERVENTION OMV", "SERVICE", "ORIGINE", "Description", "Traité par", "Cause réelle"]

# Mapping flexible : clés = variantes possibles dans les fichiers sources
COL_MAPPING = {
    "n° d'incident": "N° d'Incident",
    "n°incident": "N° d'Incident",
    "numero incident": "N° d'Incident",
    "n° incident": "N° d'Incident",
    "intervention omv": "INTERVENTION OMV",
    "intervention": "INTERVENTION OMV",
    "omv": "INTERVENTION OMV",
    "service": "SERVICE",
    "origine": "ORIGINE",
    "description": "Description",
    "traite par": "Traité par",
    "traité par": "Traité par",
    "traite_par": "Traité par",
    "cause reelle": "Cause réelle",
    "cause réelle": "Cause réelle",
    "cause_reelle": "Cause réelle"
}

all_dfs = []


def normalize_colname(name):
    if not isinstance(name, str):
        return ""
    normalized = unicodedata.normalize("NFKD", name.strip().lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = "".join(ch for ch in normalized if ch.isalnum() or ch.isspace())
    return " ".join(normalized.split())


def dedupe_columns_prefer_exact(df, raw_columns):
    new_columns = list(df.columns)
    keep_idxs = []
    seen = set()
    for i, col in enumerate(new_columns):
        if col in seen:
            continue
        seen.add(col)
        idxs = [j for j, c in enumerate(new_columns) if c == col]
        if len(idxs) == 1:
            keep_idxs.append(i)
            continue
        preferred = None
        for j in idxs:
            if raw_columns[j].strip() == col:
                preferred = j
                break
        keep_idxs.append(preferred if preferred is not None else idxs[0])
    return df.iloc[:, sorted(keep_idxs)]


xlsx_files = sorted(Path(input_folder).glob("*.xlsx"))
csv_files = sorted(Path(input_folder).glob("*.csv"))
if csv_files:
    files = csv_files
else:
    files = xlsx_files

for f in files:
    try:
        if f.suffix.lower() == ".xlsx":
            df = pd.read_excel(f, sheet_name="TOUS LES INCIDENTS", dtype=str)
        else:
            df = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    except Exception as e:
        print(f"[ERREUR] {f.name} : {e}")
        continue

    # Normaliser les noms de colonnes pour le matching
    raw_columns = list(df.columns)
    df.columns = df.columns.str.strip()
    df.columns = [c.upper() if c.strip().upper() in [t.upper() for t in TARGET_COLS] else c for c in df.columns]

    normalized_mapping = {normalize_colname(k): v for k, v in COL_MAPPING.items()}
    rename_map = {}
    for col in df.columns:
        key = normalize_colname(col)
        if key in normalized_mapping:
            rename_map[col] = normalized_mapping[key]
        elif 'traite' in key and 'par' in key:
            rename_map[col] = 'Traité par'
        elif 'cause' in key and 'reelle' in key:
            rename_map[col] = 'Cause réelle'
    df = df.rename(columns=rename_map)

    if df.columns.duplicated().any():
        df = dedupe_columns_prefer_exact(df, raw_columns)

    # Remplacer les valeurs manquantes par une chaîne vide
    df = df.fillna("")

    # Garder uniquement les colonnes cibles présentes, compléter les manquantes
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[TARGET_COLS]
    df["_source"] = f.name  # colonne optionnelle pour traçabilité
    all_dfs.append(df)
    print(f"[OK] {f.name} — {len(df)} lignes")

if all_dfs:
    result = pd.concat(all_dfs, ignore_index=True)
    output_path = "incidents_consolides.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nFichier généré : {output_path} ({len(result)} lignes total)")
else:
    print("Aucun fichier traité.")


