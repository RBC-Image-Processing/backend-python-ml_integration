# api/error_handlers.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

# Custom error handler for 404 Not Found
async def not_found_handler(request: Request, exc: HTTPException):
    logger.error(f"Path not found: {request.url}")
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )

# Custom error handler for 500 Internal Server Error
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

# Custom error handler for validation errors (422 Unprocessable Entity)
async def validation_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": exc.errors()}
    )
