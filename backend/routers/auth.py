import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from model.schemas import TokenResponse, UserLogin, UserPublic, UserRegister

router = APIRouter(prefix="/auth", tags=["Authentication"])

SECRET_KEY = os.getenv("JWT_SECRET", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))

http_bearer = HTTPBearer(auto_error=False)

_users: dict[str, dict] = {}


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(sub: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": sub, "exp": expire},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(http_bearer),
) -> UserPublic:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None or email not in _users:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    u = _users[email]
    return UserPublic(name=u["name"], email=email)


@router.post("/register", response_model=TokenResponse)
async def register(body: UserRegister):
    if body.email in _users:
        raise HTTPException(status_code=400, detail="Email already registered")
    _users[body.email] = {
        "name": body.name.strip(),
        "password_hash": _hash_password(body.password),
    }
    token = create_access_token(body.email)
    return TokenResponse(
        access_token=token,
        user=UserPublic(name=_users[body.email]["name"], email=body.email),
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    u = _users.get(body.email)
    if not u or not _verify_password(body.password, u["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(body.email)
    return TokenResponse(
        access_token=token,
        user=UserPublic(name=u["name"], email=body.email),
    )


@router.get("/me", response_model=UserPublic)
async def me(user: UserPublic = Depends(get_current_user)):
    return user
