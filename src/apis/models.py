from sqlalchemy import Column, Integer, String
from src.apis.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    age = Column(Integer)
    password = Column(String)
    role = Column(String, default="user")
    