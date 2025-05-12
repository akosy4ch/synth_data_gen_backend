from sqlalchemy import Column, Integer, String, DateTime, JSON
from .db import Base
import datetime

class Dataset(Base):
    tablename = "datasets"
    id            = Column(Integer, primary_key=True, index=True)
    filename      = Column(String, unique=True, index=True)
    upload_time   = Column(DateTime, default=datetime.datetime.utcnow)
    config       = Column(JSON)  # e.g. which columns/models used