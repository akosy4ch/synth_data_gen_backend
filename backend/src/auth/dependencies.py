from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from src.auth.security import SECRET_KEY, ALGORITHM
from src.database.db import get_db
from src.auth.database_models import User
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")  # or "/auth/login" if prefixed

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.username == username))
    user = result.scalar_one_or_none()
    if not user:
        raise credentials_exception
    return user
