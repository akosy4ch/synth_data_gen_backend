from fastapi import APIRouter, HTTPException
from backend.src.auth.models import UserRegister, UserLogin
from backend.src.auth.security import hash_password, verify_password, create_access_token

fake_users_db = {}

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register")
def register_user(user: UserRegister):
    if user.username in fake_users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    fake_users_db[user.username] = {
        "email": user.email,
        "password": hash_password(user.password)
    }
    return {"message": "User registered successfully"}

@router.post("/login")
def login_user(credentials: UserLogin):
    user = fake_users_db.get(credentials.username)
    if not user or not verify_password(credentials.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": credentials.username})
    return {"access_token": token, "token_type": "bearer"}
