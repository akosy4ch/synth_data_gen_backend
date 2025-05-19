from sqlalchemy import Column, Integer, String, DateTime, JSON ,ForeignKey
from src.database.db import Base
import datetime
from sqlalchemy.sql import func


class Dataset(Base):
    __tablename__ = "datasets"
    id            = Column(Integer, primary_key=True, index=True)
    filename      = Column(String, unique=True, index=True)
    upload_time   = Column(DateTime, default=datetime.datetime.utcnow)
    config       = Column(JSON)  # e.g. which columns/models used

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    s3_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # "original", "synthetic", etc.
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)# optional
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())