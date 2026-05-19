"""
Serveur HTTP local pour la classification d'incidents CIRVIE.
Expose predict() de main.py sur localhost:8765.
Packagé avec PyInstaller pour les postes Windows sans Python.
"""

import sys
import os
from pathlib import Path

# Résolution des chemins que PyInstaller extrait dans sys._MEIPASS
BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))

# Forcer le répertoire de travail courant sur BASE_DIR pour que
# main.py trouve les .pkl et nom_service.json
os.chdir(BASE_DIR)

from flask import Flask, request, jsonify
from main import predict

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def do_predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = predict(
            description=data.get("description", ""),
            demandeur=data.get("demandeur", "INCONNU"),
            cause=data.get("cause", "INCONNU"),
            traite=data.get("traite", "INCONNU"),
            urgence=data.get("urgence"),
            top_n=3,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, use_reloader=False, threaded=True)
