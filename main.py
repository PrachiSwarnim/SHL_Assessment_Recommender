from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import math
import numpy as np
from model import SHLRecommender

app = FastAPI(title="SHL Assessment Recommendation API (TF-IDF + SHL Catalog)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------------------------------
# 📘 Load SHL individual test catalog (from scraper)
# -------------------------------------------------------
catalog_path = "SHL_Scraped_Assessments.csv"

# Load and normalize column names
catalog_df = pd.read_csv(catalog_path)
catalog_df.columns = [c.strip().replace(" ", "_").title() for c in catalog_df.columns]

print(f"✅ Loaded {len(catalog_df)} SHL individual tests from catalog.")

# Initialize the recommender with catalog data
recommender = SHLRecommender(catalog_df)


# -------------------------------------------------------
# 🧠 API Models
# -------------------------------------------------------
class QueryInput(BaseModel):
    query: str


# -------------------------------------------------------
# 🌐 Endpoints
# -------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(input: QueryInput):
    try:
        results = recommender.recommend(input.query, top_k=10)
        formatted_results = []
        for r in results:
            formatted_results.append({
                "url": r["url"],
                "name": r["name"],
                "adaptive_support": r.get("adaptive_irt", "No"),
                "description": r.get("description", "N/A"),
                "duration": r.get("duration", None),
                "remote_support": r.get("remote_testing", "Yes"),
                "test_type": [t.strip() for t in str(r["test_type"]).split(",") if t.strip()]
            })

        return {"recommended_assessments": formatted_results}

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}


@app.get("/metrics")
def metrics():
    try:
        with open("metrics.txt", "r") as f:
            score = f.read().strip()
        return {"Mean Recall@10": score}
    except FileNotFoundError:
        return {"Mean Recall@10": "Unavailable"}


@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("index.html")
