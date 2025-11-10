# Importing libraries
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from model import SHLRecommender
import os

# Initializing FastAPI app
app = FastAPI(title="SHL Assessment Recommendation API (TF-IDF + SHL Catalog)")

# Enabling CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shl-assessment-recommender-ellx.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading SHL catalog
catalog_path = "SHL_Scraped_Assessments.csv"

try:
    catalog_df = pd.read_csv(catalog_path)
    catalog_df.columns = [c.strip().replace(" ", "_").title() for c in catalog_df.columns]
except Exception as e:
    print(f"Failed to load catalog: {e}")
    catalog_df = pd.DataFrame()

# Initializing recommender model
recommender = SHLRecommender(catalog_df)

# Defining input schema
class QueryInput(BaseModel):
    query: str


# Health check
@app.get("/api/health")
def health_check():
    return {"status": "ok"}


# Universal preflight handler 
@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str):
    response = JSONResponse(content={"message": "Preflight OK"})
    response.headers["Access-Control-Allow-Origin"] = "https://shl-assessment-recommender-ellx.onrender.com"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# Main recommendation route
@app.post("/api/recommend")
async def recommend(input: QueryInput):
    """
    Accepts a text query and returns top SHL assessment recommendations.
    """
    try:
        if not input.query.strip():
            return {"error": "Empty query provided"}

        results = recommender.recommend(input.query, top_k=10)
        formatted_results = []

        for r in results:
            formatted_results.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "adaptive_support": r.get("adaptive_support", "N/A"),
                "description": r.get("description", "N/A"),
                "duration": r.get("duration", 0),
                "remote_support": r.get("remote_support", "N/A"),
                "test_type": [
                    t.strip() for t in str(r.get("test_type", "")).split(",") if t.strip()
                ]
            })

        return {"recommended_assessments": formatted_results}

    except Exception as e:
        print("Error in recommend:", e)
        return {"error": str(e)}


# Serve frontend HTML
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serve the main HTML directly from backend, bypassing Render's static cache.
    """
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        html_content = f.read()
    # adding cache-control header
    return HTMLResponse(content=html_content, headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    })

# Disabling caching for all responses
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Access-Control-Allow-Origin"] = "https://shl-assessment-recommender-ellx.onrender.com"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


# Mounting static directory AFTER routes to avoid route conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")
