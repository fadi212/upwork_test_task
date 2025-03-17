from sqlalchemy.orm import Session
from src.apis import models, schemas, auth
from oso import Oso
from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError

oso = Oso()
oso.register_class(models.User)
oso.load_files(["src/apis/oso_policy.polar"])


def authorize(user, action, resource, detail):
    if not oso.is_allowed(user, action, resource):
        raise HTTPException(status_code=403, detail=detail)


def create_user(db: Session, user: schemas.UserCreate):
    existing_user = db.query(models.User).filter(models.User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = models.User(
        name=user.name, 
        email=user.email, 
        password=auth.hash_password(user.password), 
        age=user.age, 
        role="user"
    )

    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Email already registered")


def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()


def get_all_users(db: Session):
    return db.query(models.User).all()


def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user_update.name is not None:
        user.name = user_update.name
    if user_update.age is not None:
        user.age = user_update.age
    if user_update.password is not None:
        user.password = auth.hash_password(user_update.password)    

    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user_id: int):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}
