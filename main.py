import os
from datetime import timedelta
from fastapi import FastAPI, Depends, Form, UploadFile, HTTPException, File, Body
from sqlalchemy.orm import Session
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer
from src.apis import models
from src.apis import schemas
from src.apis import crud
from src.apis import database
from src.apis import auth
from src.ml.index import upload_and_index
from src.ml.query.query import query_index_async

app = FastAPI()

# Custom JWT auth settings for swager
security = HTTPBearer()
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="My API",
        version="1.0.0",
        description="JWT Authentication Example",
        routes=app.routes,
    )

    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path:
            path[method]["security"] = [{"BearerAuth": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema
app.openapi = custom_openapi


# Database config
models.Base.metadata.create_all(bind=database.engine)
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Routes
@app.post("/register/", response_model=schemas.UserResponse)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    return crud.create_user(db, user)


@app.post("/login")
def login(data: schemas.LoginRequest, db: Session = Depends(get_db)):
    user = auth.authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = auth.create_access_token(
        data={"sub": str(user.id)}, expires_delta=timedelta(minutes=30))
    return {"access_token": access_token}


@app.get("/users/", response_model=list[schemas.UserResponse])
def get_all_users(current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    crud.authorize(current_user, "get_all_users", "User",
                   detail="Permission denied, only admin can get all users.")
    return crud.get_all_users(db)


@app.get("/users/user-role")
def get_all_users_with_roles(current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Retrieve all users and their roles (Only accessible to admins)."""
    crud.authorize(current_user, "get_all_users_with_role", "User",
                   detail="Permission denied, only admins can get users and their roles.")
    users = db.query(models.User).all()
    return [{"id": user.id, "email": user.email, "role": user.role} for user in users]


@app.get("/users/{user_id}", response_model=schemas.UserResponse)
def get_profile(user_id: int, current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    crud.authorize(current_user, "view_profile", user,
                   detail="Permission denied, you can get only your own profile.")
    return user


@app.put("/users/edit-profile", response_model=schemas.UserResponse)
def edit_profile(user_update: schemas.UserUpdateWithEmail, current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    
    user = crud.get_user_by_email(db, user_update.email)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    crud.authorize(current_user, "edit_profile", user, detail="Permission denied. you are not admin, can only edit your own profile")

    return crud.update_user(db, user.id, user_update)


@app.delete("/users/{user_id}")
def delete_user(user_id: int, current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    user = crud.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    crud.authorize(current_user, "delete_user", user,
                   detail="Permission denied, only admins can delete users.")
    return crud.delete_user(db, user_id)


@app.post("/users/update-role/")
def update_role(request: schemas.RoleAssignmentRequest, current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    """Allow admins to assign roles to users."""

    crud.authorize(current_user, "update_role", "User",
                   detail="Permission denied, only admins can update user's role.")

    user = db.query(models.User).filter(
        models.User.email == request.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if request.role not in ["admin", "user"]:
        raise HTTPException(
            status_code=400, detail="Invalid role. Type 'admin' or 'user'.")

    user.role = request.role
    db.commit()

    return {"message": f'{request.role} role assigned to {user.email}'}


# ML side APIs
@app.post("/upload-and-index/")
async def upload_and_index_endpoint(file: UploadFile = File(...), current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):

    try:
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)

        local_file_path = os.path.join(upload_dir, file.filename)
        with open(local_file_path, "wb") as f_out:
            f_out.write(await file.read())

        result = upload_and_index(
            local_file_path=local_file_path
        )

        if result.get("message"):
            return {
                "status": "success",
                "message": result["message"],
                "details": result
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Indexing pipeline returned an unexpected response: {result}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/query/")
async def query_endpoint(query: str = Form(...), session_id: str = Form(None), current_user: schemas.UserResponse = Depends(auth.get_current_user), db: Session = Depends(get_db)):

    if not session_id:
        session_id = "default"

    try:
        result = await query_index_async(query_text=query, session_id=session_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "response": result["response"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uvicorn main:app --reload
