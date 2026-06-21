"""
Serveur HTTP local pour la classification d'incidents CIRVIE.
Expose predict() de main.py sur localhost:8765.
Packagé avec PyInstaller pour les postes Windows sans Python.
"""

import sys
import os
from pathlib import Path

# Quand on tourne depuis un exe PyInstaller --onefile, les .pkl et nom_service.json
# sont dans le même dossier que l'exe (sys.executable), pas dans sys._MEIPASS.
# En développement normal, ils sont à côté de ce fichier.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

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
    if not data.get("description"):
        return jsonify({"error": "description is required"}), 400
    try:
        result = predict(
            description=data.get("description", ""),
            demandeur=data.get("demandeur", "INCONNU"),
            cause=data.get("cause", "INCONNU"),
            traite=data.get("traite", "INCONNU"),
            urgence=data.get("urgence"),
            top_n=3,
            confidence_threshold=data.get("confidence_threshold", 0.0),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8765, use_reloader=False, threaded=True)
