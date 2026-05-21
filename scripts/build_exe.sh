#!/usr/bin/env bash
# ============================================================
#  Build predict_server (Mac / Linux / CI)
#  Note : l'exe produit ici est natif macOS/Linux.
#  Pour un exe Windows, compiler sur Windows avec build_exe.bat
# ============================================================
set -e

pip install --quiet pyinstaller flask

pyinstaller --onefile --noconsole \
  --name predict_server \
  --add-data "models/model_fast_omv.pkl:." \
  --add-data "models/model_fast_service.pkl:." \
  --add-data "models/model_fast_origine.pkl:." \
  --add-data "models/nom_service.json:." \
  predict_server.py

echo ""
echo "[OK] Binaire généré : dist/predict_server"
