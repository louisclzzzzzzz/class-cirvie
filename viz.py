"""Persistance de matrices de confusion pour les modèles ML (OMV / ORIGINE)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_confusion_matrix(result: dict[str, Any], label: str, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_classes = len(result["label_encoder"].classes_)
    fig_size = max(6, n_classes * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    ConfusionMatrixDisplay.from_predictions(
        result["y_true"],
        result["y_pred"],
        normalize="true",
        xticks_rotation=45,
        ax=ax,
        colorbar=True,
    )
    ax.set_title(f"Matrice de confusion — {label}")
    fig.tight_layout()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"confusion_matrix_{label}_{timestamp}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
