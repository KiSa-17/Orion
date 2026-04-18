import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter

from model.schemas import (
    HospitalCapacityBody,
    HospitalCapacityOut,
    PatientFlowInput,
    PatientFlowOutput,
    ReadmissionInput,
    ReadmissionOutput,
)
from models.loader import get_model, get_ts_regressor, get_ts_scaler, snapshot_loaded_model_keys

router = APIRouter(prefix="/predict", tags=["Predictions"])


def _patient_flow_heuristic(data: PatientFlowInput) -> int:
    base = 28 + data.hour_of_day * 2 + (data.month % 4) * 3
    base += (data.day_of_week % 3) * 4
    if data.is_holiday:
        base -= 18
    if data.avg_temp is not None and data.avg_temp > 35:
        base += 12
    return max(5, int(base))


def _readmission_heuristic(data: ReadmissionInput) -> float:
    score = (
        0.08
        + min(0.35, data.length_of_stay * 0.04)
        + min(0.25, data.num_medications * 0.015)
        + min(0.15, data.num_diagnoses * 0.02)
        + min(0.12, data.num_prior_visits * 0.03)
        + min(0.1, max(0, (data.age - 40)) * 0.002)
    )
    return min(0.97, float(score))


@router.post("/patient-flow", response_model=PatientFlowOutput)
async def predict_patient_flow(data: PatientFlowInput):
    try:
        model = get_model("patient_flow")
    except KeyError:
        model = None

    if model is not None:
        features = np.array(
            [
                [
                    data.day_of_week,
                    data.hour_of_day,
                    data.month,
                    data.is_holiday,
                    data.avg_temp if data.avg_temp is not None else 25.0,
                ]
            ]
        )
        prediction = model.predict(features)
        predicted_count = int(prediction[0])
        return PatientFlowOutput(
            predicted_patients=predicted_count,
            model_used="patient_flow",
            confidence_note="Based on trained patient-flow model",
        )

    predicted_count = _patient_flow_heuristic(data)
    return PatientFlowOutput(
        predicted_patients=predicted_count,
        model_used="heuristic_fallback",
        confidence_note="No patient_flow .pkl found — using a demo heuristic until MODEL_DIR contains the trained file",
    )


@router.post("/readmission-risk", response_model=ReadmissionOutput)
async def predict_readmission(data: ReadmissionInput):
    try:
        model = get_model("readmission")
    except KeyError:
        model = None

    features = np.array(
        [
            [
                data.age,
                data.length_of_stay,
                data.num_diagnoses,
                data.num_medications,
                data.num_prior_visits,
            ]
        ]
    )

    if model is not None:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(features)[0][1])
        else:
            prob = float(model.predict(features)[0])
    else:
        prob = _readmission_heuristic(data)

    if prob < 0.3:
        level, recommendation = "Low", "Standard discharge process."
    elif prob < 0.6:
        level, recommendation = "Medium", "Schedule follow-up within 7 days."
    else:
        level, recommendation = "High", "Arrange home care and 48-hour follow-up call."

    return ReadmissionOutput(
        risk_score=round(prob, 4),
        risk_level=level,
        recommendation=recommendation,
        model_used="readmission" if model is not None else "heuristic_fallback",
    )


@lru_cache
def _ts_feature_columns() -> tuple[str, ...]:
    p = Path(__file__).resolve().parent.parent / "model" / "feature_columns_ts.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return tuple(str(x) for x in data)


def _default_ts_feature(col: str) -> float:
    defaults: dict[str, float] = {
        "District": 5.0,
        "City": 12.0,
        "City_Type": 1.0,
        "ICU_Beds": 12.0,
        "Non_ICU_Beds": 80.0,
        "Available_Beds": 55.0,
        "ICU_Rooms": 8.0,
        "General_Ward_Rooms": 24.0,
        "Private_Rooms": 10.0,
        "Semi_Private_Rooms": 8.0,
        "Emergency_Rooms": 4.0,
        "Operation_Theatres": 3.0,
        "Isolation_Rooms": 2.0,
        "Total_Rooms": 60.0,
        "Ventilators": 10.0,
        "Oxygen_Concentrators": 15.0,
        "ECG_Machines": 6.0,
        "Defibrillators": 5.0,
        "Pulse_Oximeters": 20.0,
        "CT_Scan": 1.0,
        "X_Ray_Machines": 2.0,
        "Ultrasound_Units": 2.0,
        "Blood_Analyzers": 2.0,
        "Patient_Monitors": 12.0,
        "Doctors_General_Physician": 6.0,
        "Doctors_Pulmonologist": 2.0,
        "Doctors_Cardiologist": 2.0,
        "Doctors_Intensivist": 2.0,
        "Doctors_Anesthesiologist": 2.0,
        "Doctors_Surgeon": 3.0,
        "Doctors_Radiologist": 2.0,
        "Doctors_Neurologist": 1.0,
        "Doctors_Orthopedic": 2.0,
        "Doctors_Pediatrician": 2.0,
        "Total_Doctors": 28.0,
        "wait_minutes": 32.0,
        "day_of_week": 2.0,
        "month_num": 6.0,
        "is_holiday": 0.0,
        "day_trend": 0.0,
    }
    return float(defaults.get(col, 0.0))


def _heuristic_total_beds(feat: dict[str, float], _cols: tuple[str, ...]) -> float:
    def g(c: str) -> float:
        if c in feat:
            return float(feat[c])
        return _default_ts_feature(c)

    icu = g("ICU_Beds")
    non = g("Non_ICU_Beds")
    avail = g("Available_Beds")
    wait = g("wait_minutes")
    docs = g("Total_Doctors")
    rooms = g("Total_Rooms")
    dow = g("day_of_week")
    hol = g("is_holiday")
    out = icu * 2.2 + non * 1.15 + rooms * 0.4 + docs * 1.1
    out += wait * 0.12 + dow * 1.5 - hol * 12.0 + max(0.0, 85.0 - avail) * 0.35
    return float(max(12.0, min(800.0, out)))


def _build_ts_row(body: HospitalCapacityBody, cols: tuple[str, ...]) -> np.ndarray:
    row = []
    f = body.features or {}
    for c in cols:
        if c in f:
            row.append(float(f[c]))
        else:
            row.append(_default_ts_feature(c))
    return np.array([row], dtype=np.float64)


@router.post("/hospital-capacity", response_model=HospitalCapacityOut)
async def predict_hospital_capacity(body: HospitalCapacityBody):
    cols = _ts_feature_columns()
    key, model = get_ts_regressor()
    scaler = get_ts_scaler()
    X = _build_ts_row(body, cols)

    if model is not None:
        try:
            X_in = scaler.transform(X) if scaler is not None else X
            n_feat = int(getattr(model, "n_features_in_", X_in.shape[1]))
            if X_in.shape[1] != n_feat:
                raise ValueError(f"feature count mismatch: got {X_in.shape[1]} need {n_feat}")
            pred = float(model.predict(X_in)[0])
            return HospitalCapacityOut(
                predicted_total_beds=round(pred, 2),
                model_used=key or "ts_regressor",
                notes="Loaded from ml models 2 artefacts (model_*.pkl / scaler.pkl).",
            )
        except Exception:
            pass

    pred = _heuristic_total_beds(body.features or {}, cols)
    return HospitalCapacityOut(
        predicted_total_beds=round(pred, 2),
        model_used="heuristic_fallback",
        notes="Train pipeline in ml models 2 and save model_*.pkl + scaler.pkl to enable sklearn inference.",
    )


@router.get("/hospital-capacity/schema")
async def hospital_capacity_schema():
    return {"feature_columns": list(_ts_feature_columns())}


@router.get("/ml-models-2/status")
async def ml_models_2_status():
    snap = snapshot_loaded_model_keys()
    key, _ = get_ts_regressor()
    return {
        "feature_column_count": len(_ts_feature_columns()),
        "active_ts_regressor": key,
        "has_ts_scaler": snap["has_ts_scaler"],
        "loaded_ml_models_2_keys": snap["ml_models_2_regressors"],
        "endpoints": [
            {"path": "/predict/patient-flow", "method": "POST"},
            {"path": "/predict/readmission-risk", "method": "POST"},
            {"path": "/predict/hospital-capacity", "method": "POST"},
            {"path": "/predict/hospital-capacity/schema", "method": "GET"},
        ],
    }
