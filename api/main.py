# api/main.py
import uvicorn
from api.api_endpoints import app

if __name__ == "__main__":
    # Start the FastAPI app with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)