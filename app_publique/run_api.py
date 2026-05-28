"""Launch script for the FastAPI auth server."""

import uvicorn
from api.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
