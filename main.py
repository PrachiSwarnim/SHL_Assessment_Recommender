# Importing libraries
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
from model import SHLRecommender
from fastapi.staticfiles import StaticFiles
import os
from fastapi import Request

# Initializing FastAPI application
app = FastAPI(title="SHL Assessment Recommendation API (TF-IDF + SHL Catalog)")

# Enabling Cross-Origin Resource Sharing (CORS)
# This ensures that frontend (HTML/JS) hosted elsewhere can make API requests safely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://shl-assessment-recommender-ellx.onrender.com"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loading and preparing the SHL catalog
catalog_path = "SHL_Scraped_Assessments.csv"

try:
    catalog_df = pd.read_csv(catalog_path)
    # Normalizing column names (Capitalization and removing spaces)
    catalog_df.columns = [c.strip().replace(" ", "_").title() for c in catalog_df.columns]
except Exception as e:
    # Handling missing or unreadable catalog file
    print(f"Failed to load catalog: {e}")
    catalog_df = pd.DataFrame()

# Initializing the recommender model using the loaded dataset
recommender = SHLRecommender(catalog_df)

# Defining input data structure
class QueryInput(BaseModel):
    query: str

# API Endpoints
@app.get("/health")
def health_check():
    """Simple heartbeat endpoint to verify the API is running."""
    return {"status": "ok"}


@app.post("/recommend")
def recommend(input: QueryInput):
    """
    Accepts a text query and returns
    the top SHL assessment recommendations ranked by similarity.
    """
    try:
        if not input.query.strip():
            return {"error": "Empty query provided"}

        # Generating top 10 recommendations from the model
        results = recommender.recommend(input.query, top_k=10)
        formatted_results = []

        # Cleaning and structuring output for frontend compatibility
        for r in results:
            formatted_results.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "test_type": [
                    t.strip() for t in str(r.get("test_type", "")).split(",") if t.strip()
                ]
            })

        return {"recommended_assessments": formatted_results}

    except Exception as e:
        # Logging unexpected errors without crashing the app
        print("Error in recommend:", e)
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """
    Serves the main HTML interface for the application.
    This lets users type a job query and see assessment results visually.
    """
    return FileResponse(os.path.join("static", "index.html"))

@app.options("/recommend")
async def options_recommend(request: Request):
    return {"status": "ok"}
    
app.mount("/static",
StaticFiles(directory="static"),
name="static")