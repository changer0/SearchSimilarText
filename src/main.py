# app.py
import uvicorn
from fastapi import FastAPI
app = FastAPI()

# main.py

if __name__ == '__main__':
    uvicorn.run(app="app:app", port=8000, reload=True)
