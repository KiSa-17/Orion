import glob
import joblib
import os

from config import settings

_models: dict = {}


def load_all_models():
    """
    Called once at app startup.
    Add every model you want to expose here.
    """
    model_files = {
        "maharastra_bundle": "Maharastra_model_bundle.pkl",
        # "patient_flow": "patient_flow_model.pkl",
        # "readmission": "readmission_model.pkl",
    }

    for name, filename in model_files.items():
        path = os.path.join(settings.MODEL_DIR, filename)
        if os.path.exists(path):
            loaded = joblib.load(path)
            if isinstance(loaded, dict):
                for key, val in loaded.items():
                    if hasattr(val, "predict") or hasattr(val, "predict_proba"):
                        _models[key] = val
                _models[name] = loaded
            else:
                _models[name] = loaded
            print(f"[ok] Loaded model: {name} from {path}")
        else:
            print(f"[skip] Model file not found: {path}")

    _load_models2_artifacts()


def _load_models2_artifacts() -> None:
    """Optional sklearn models + scaler from ml models 2 (pipeline outputs)."""
    m2 = getattr(settings, "MODELS_2_DIR", "") or ""
    if not m2 or not os.path.isdir(m2):
        return
    for p in sorted(glob.glob(os.path.join(m2, "model_*.pkl"))):
        base = os.path.basename(p).replace(".pkl", "").replace("model_", "ts_")
        try:
            obj = joblib.load(p)
            if hasattr(obj, "predict"):
                _models[base] = obj
                print(f"[ok] models2 regressor: {base} <- {p}")
        except Exception as exc:
            print(f"[skip] models2 load {p}: {exc}")
    sp = os.path.join(m2, "scaler.pkl")
    if os.path.isfile(sp):
        try:
            _models["ts_scaler"] = joblib.load(sp)
            print(f"[ok] models2 scaler: {sp}")
        except Exception as exc:
            print(f"[skip] models2 scaler: {exc}")


def get_model(name: str):
    if name not in _models:
        raise KeyError(f"Model '{name}' is not loaded. Check models/loader.py and your model files.")
    return _models[name]


def get_ts_scaler():
    return _models.get("ts_scaler")


def get_ts_regressor():
    """Prefer HistGradBoost / RandomForest artefact from ml models 2 folder."""
    for key in ("ts_histgradboost", "ts_randomforest", "ts_lightgbm", "ts_catboost"):
        if key in _models:
            return key, _models[key]
    for key, val in _models.items():
        if key.startswith("ts_") and key != "ts_scaler" and hasattr(val, "predict"):
            return key, val
    return None, None


def snapshot_loaded_model_keys() -> dict[str, list[str] | bool]:
    return {
        "all_models": sorted(_models.keys()),
        "ml_models_2_regressors": sorted(
            k for k in _models if k.startswith("ts_") and k != "ts_scaler"
        ),
        "has_ts_scaler": "ts_scaler" in _models,
    }
