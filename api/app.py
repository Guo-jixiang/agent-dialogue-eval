from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.routes import router

app = FastAPI(title="对话模型评测系统", version="1.0.0")
app.include_router(router, prefix="/api")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: log the full traceback and return a JSON 500 with the error message.
    Note: HTTPException (including 404) is handled by FastAPI's default handler.
    """
    # Don't intercept HTTPExceptions — let FastAPI handle 404/422 etc. normally
    if isinstance(exc, HTTPException):
        raise exc
    logger.exception(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Web dist directory (built by `npm run build` inside web/)
WEB_DIST = Path(__file__).parent.parent / "web" / "dist"
# Fallback: legacy single-file index.html during development
WEB_LEGACY = Path(__file__).parent.parent / "web" / "index.html"


def _spa_index() -> FileResponse:
    """Return the SPA entry point (built dist or legacy single-file)."""
    if (WEB_DIST / "index.html").exists():
        return FileResponse(WEB_DIST / "index.html")
    return FileResponse(WEB_LEGACY)


# Mount built static assets if dist/ exists
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(WEB_DIST / "assets")), name="assets")


# SPA catch-all: serve index.html for any non-API path so Vue Router works
@app.get("/")
async def root():
    return _spa_index()


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    """Serve SPA index for all non-API, non-asset routes."""
    # Don't intercept API routes (already handled above)
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404)
    # Serve real static files from dist if they exist
    if WEB_DIST.exists():
        candidate = WEB_DIST / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
    return _spa_index()
