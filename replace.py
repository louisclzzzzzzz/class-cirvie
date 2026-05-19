import pandas as pd

df = pd.read_csv("data3.csv", encoding="utf-8-sig")

replacements = {
    "SER I":  "SER IND PARIS",
    "SERI":   "SER IND PARIS",
    "SERC":   "SER COLL PARIS",
}

df = df.replace(replacements)

df.to_csv("data3.csv", index=False, encoding="utf-8-sig")
print("✓ Remplacement effectué dans data3.csv")