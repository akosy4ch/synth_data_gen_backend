# src/database/utils.py

from src.database.models import UploadedFile
from src.database.db import get_db

async def save_file_record_to_db(
    filename: str,
    s3_path: str,
    file_type: str,
    owner_id: int = None
):
    async for session in get_db():
        new_file = UploadedFile(
            filename=filename,
            s3_path=s3_path,
            file_type=file_type,
            owner_id=owner_id
        )
        session.add(new_file)
        await session.commit()
        break
