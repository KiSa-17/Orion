# backend/routers/resources.py
import random

from fastapi import APIRouter

from model.schemas import ResourceStatus

router = APIRouter(prefix="/resources", tags=["Hospital Resources"])

@router.get("/status", response_model=ResourceStatus)
async def get_resource_status():
    """
    Returns current hospital resource availability.
    In production: replace with a real DB query.
    """
    beds_total = 200
    beds_occupied = 135 + random.randint(-8, 12)
    beds_occupied = max(80, min(beds_total - 20, beds_occupied))
    beds_available = beds_total - beds_occupied
    doctors = 16 + random.randint(0, 4)
    nurses = 38 + random.randint(0, 6)
    icu = max(2, 8 - (beds_occupied // 40))
    status = "Busy" if beds_occupied > 150 else "Optimal"
    return ResourceStatus(
        beds_total=beds_total,
        beds_occupied=beds_occupied,
        beds_available=beds_available,
        doctors_on_duty=doctors,
        nurses_on_duty=nurses,
        icu_beds_available=icu,
        status=status,
    )