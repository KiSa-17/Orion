# backend/model/schemas.py
from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Any, Optional


# ── Auth ─────────────────────────────────────────────────────────────
class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    name: str
    email: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic

# ── Patient Flow Prediction ──────────────────────────────────────────
class PatientFlowInput(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday, 6=Sunday")
    hour_of_day: int = Field(..., ge=0, le=23)
    month: int       = Field(..., ge=1, le=12)
    is_holiday: int  = Field(0, ge=0, le=1)
    avg_temp: Optional[float] = Field(None, description="Outside temperature (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "day_of_week": 1,
                "hour_of_day": 14,
                "month": 6,
                "is_holiday": 0,
                "avg_temp": 32.5
            }
        }

class PatientFlowOutput(BaseModel):
    predicted_patients: int
    model_used: str
    confidence_note: str

# ── Readmission Risk Prediction ───────────────────────────────────────
class ReadmissionInput(BaseModel):
    age: int               = Field(..., ge=0, le=120)
    length_of_stay: int    = Field(..., ge=0, description="Days in hospital")
    num_diagnoses: int     = Field(..., ge=1)
    num_medications: int   = Field(..., ge=0)
    num_prior_visits: int  = Field(0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 65,
                "length_of_stay": 5,
                "num_diagnoses": 3,
                "num_medications": 8,
                "num_prior_visits": 2
            }
        }

class ReadmissionOutput(BaseModel):
    risk_score: float        # Raw probability 0.0 - 1.0
    risk_level: str          # "Low", "Medium", "High"
    recommendation: str
    model_used: str = "readmission"

# ── Resource Status ───────────────────────────────────────────────────
class ResourceStatus(BaseModel):
    beds_total: int
    beds_occupied: int
    beds_available: int
    doctors_on_duty: int
    nurses_on_duty: int
    icu_beds_available: int
    status: str              # "Optimal", "Busy", "Critical"


# ── Hospital capacity (ml models 2 time-series feature set) ───────────
class HospitalCapacityBody(BaseModel):
    """Partial feature map; missing keys use server defaults for inference."""

    features: dict[str, float] = Field(default_factory=dict)

    @field_validator("features", mode="before")
    @classmethod
    def _coerce_feature_map(cls, v: Any) -> dict[str, float]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise TypeError("features must be a JSON object")
        out: dict[str, float] = {}
        for key, val in v.items():
            try:
                out[str(key)] = float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature '{key}' must be numeric") from exc
        return out


class HospitalCapacityOut(BaseModel):
    predicted_total_beds: float
    model_used: str
    notes: str = ""