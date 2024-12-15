# app.py
import uvicorn
from fastapi import FastAPI
app = FastAPI()

# main.py
from app import app

if __name__ == '__main__':
    uvicorn.run(app="app:app", port=8000, reload=True)
