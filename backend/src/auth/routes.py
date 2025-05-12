from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext


from src.auth.security import hash_password, verify_password, create_access_token
from src.db import get_db  # updated import
from src.auth.database_models import User
from src.auth.database_models import UserRegister, UserLogin  # your ORM User model

router = APIRouter(prefix="/auth", tags=["Auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register")
async def register_user(
    user: UserRegister,
    db: AsyncSession = Depends(get_db),  # use AsyncSession via get_db
):
    # Check if username already exists
    result = await db.execute(select(User).where(User.username == user.username))
    existing_user = result.scalar_one_or_none()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")

    hashed_password = pwd_context.hash(user.password)

    # Create new user
    new_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
    )
    db.add(new_user)
    await db.commit()       # async commit
    await db.refresh(new_user)  # refresh instance
    return {"message": "User registered successfully"}

@router.post("/login")
async def login_user(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),  # use AsyncSession via get_db
):
    # Fetch user by username
    result = await db.execute(select(User).where(User.username == credentials.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")


    # Create JWT token
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}
