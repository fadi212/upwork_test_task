from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    age: int
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    password: Optional[str] = None

class UserUpdateWithEmail(BaseModel):
    email: str
    name: Optional[str] = None
    age: Optional[int] = None
    password: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: int
    role: str

    class Config:
        from_attributes = True

class RoleAssignmentRequest(BaseModel):
    email: EmailStr
    role: str
