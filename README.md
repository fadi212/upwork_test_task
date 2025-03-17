# User Crud AI Project Overview

This project is a **User Crud** built with **FastAPI**, providing complete **CRUD operations for user management**, secure **authentication**, and **role-based authorization** using the **Oso framework**. It ensures access control, allowing different user roles (e.g., admin, user) to perform specific actions securely.  

### 🔹 Key Features:  

- **User Registration & Authentication:** Secure login and registration using **JWT-based authentication**.  
- **Role-Based Access Control (RBAC):** Authorization powered by **Oso**, enforcing access based on user roles.  
- **CRUD Operations:** Manage user profiles with **create, read, update, and delete** functionalities.  
- **Admin Controls:** Admins can **view, modify, and delete** users and assign roles dynamically.  
- **Secure API Endpoints:** All endpoints are protected, ensuring that users can only access permitted resources.  
- **ML Service Integration (Optional):** Supports document indexing and querying.

---

# Project Setup

## ⚙️ Prerequisites

Before setting up the project, ensure you have the following installed:

- [Python](https://www.python.org/downloads/) (version 3.8+ recommended)
- [PostgreSQL](https://www.postgresql.org/download/) (using PostgreSQL as the database)

---

## Setup Instructions

### 🔹 Create a Virtual Environment

Open a terminal or command prompt in your project directory and run:

```sh
python -m venv venv
```

This will create a `venv` folder containing the virtual environment.

---

### 🔹 Activate the Virtual Environment

Run the following command on terminal:

#### Command Prompt (cmd.exe):

```sh
venv\Scripts\activate
```

### 🔹 Install Dependencies

Once the virtual environment is activated, install project dependencies:

```sh
pip install -r requirements.txt
```

---

### 🔹 Set Up Environment Variables for database

Open `.env` file from the project's root and modify the database variable:

```ini
DATABASE_URL=postgresql://<username>:<password>@<host (localhost in case of running locally)>:<port>/<database name>
```

Replace the placeholder values with your actual configuration.

---

### 🔹 Start the FastAPI Server

Run the following command to start the server:

```sh
uvicorn main:app --reload
```

By default, the API will be available at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

### 🔹 Access API Documentation

Once the server is running, you can access the API documentation:

- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

# API Overview

### 🔹 Authentication APIs

#### **Register a new user**

This endpoint will register new user in database

```http
POST /register/
```
- Request Body: `name`, `email`, `age`, `password`

#### **Login**
```http
POST /login/
```
- Request Body: `email`, `password`
- Response: JWT access token

### User Management APIs

#### **Get all users (Admin only access)**
```http
GET /users/
```
- Requires authentication (Admin role)

#### **Get specific user detail by id**
```http
GET /users/{user_id}
```
- Users can only access their own profile.
- Admins can access profiles of all users.

#### **Edit Profile**
```http
PUT /users/edit-profile/
```
- Email will be use as a unique identifier for editing
- Users can edit their own profile
- Admins can edit any profile

#### **Delete User (Admin only)**
```http
DELETE /users/{user_id}
```
- Admins can delete any user

#### **Update User Role (Admin only)**
```http
POST /users/update-role/
```
- Admins can assign and update user roles.

### 🔹 ML APIs

#### **Upload and Index a File(pdf)**
```http
POST /upload-and-index/
```
- Uploads a file and indexes it for querying

#### **Query Indexed Data**
```http
POST /query/
```
- Allows querying indexed data

---

## Stopping the Server

To stop the running FastAPI server, press:

```sh
CTRL + C
```

To deactivate the virtual environment, run:

```sh
deactivate
```

---
