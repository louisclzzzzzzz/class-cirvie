import main as main_module


def _omv_subset(cleaned_synthetic_df):
    return cleaned_synthetic_df[
        cleaned_synthetic_df["OMV"].notna() & (cleaned_synthetic_df["Description"] != "")
    ]


def _origine_subset(cleaned_synthetic_df):
    return cleaned_synthetic_df[
        cleaned_synthetic_df["ORIGINE_clean"].notna()
        & (cleaned_synthetic_df["OMV"] == "OUI")
        & (cleaned_synthetic_df["Description"] != "")
    ]


def test_kfold_cv_omv_mean_accuracy(cleaned_synthetic_df):
    result = main_module.run_cross_validation(
        _omv_subset(cleaned_synthetic_df), "OMV", "OMV", main_module.TABULAR_CONFIG["OMV"], n_splits=5
    )
    assert result["mean_accuracy"] >= 0.70


def test_kfold_cv_omv_std_below_threshold(cleaned_synthetic_df):
    result = main_module.run_cross_validation(
        _omv_subset(cleaned_synthetic_df), "OMV", "OMV", main_module.TABULAR_CONFIG["OMV"], n_splits=5
    )
    assert result["std_accuracy"] <= 0.15


def test_kfold_cv_origine_mean_f1(cleaned_synthetic_df):
    result = main_module.run_cross_validation(
        _origine_subset(cleaned_synthetic_df), "ORIGINE_clean", "ORIGINE", main_module.TABULAR_CONFIG["ORIGINE"], n_splits=5
    )
    assert result["mean_f1"] >= 0.40


def test_kfold_cv_returns_expected_keys(cleaned_synthetic_df):
    result = main_module.run_cross_validation(
        _omv_subset(cleaned_synthetic_df), "OMV", "OMV", main_module.TABULAR_CONFIG["OMV"], n_splits=5
    )
    assert {"mean_accuracy", "std_accuracy", "mean_f1", "std_f1", "scores", "n_splits"}.issubset(result.keys())
