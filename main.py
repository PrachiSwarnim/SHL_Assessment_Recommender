from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
from model import SHLRecommender

# -------------------------------------------------------
# 🚀 Initialize FastAPI app
# -------------------------------------------------------
app = FastAPI(title="SHL Assessment Recommendation API (TF-IDF + SHL Catalog)")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace '*' with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# 📘 Load SHL individual test catalog (scraped dataset)
# -------------------------------------------------------
catalog_path = "SHL_Scraped_Assessments.csv"

try:
    catalog_df = pd.read_csv(catalog_path)
    catalog_df.columns = [c.strip().replace(" ", "_").title() for c in catalog_df.columns]
    print(f"✅ Loaded {len(catalog_df)} SHL individual tests from catalog.")
except Exception as e:
    print(f"❌ Failed to load catalog: {e}")
    catalog_df = pd.DataFrame()

# Initialize recommender model
recommender = SHLRecommender(catalog_df)

# -------------------------------------------------------
# 🧠 API Input Model
# -------------------------------------------------------
class QueryInput(BaseModel):
    query: str

# -------------------------------------------------------
# 🌐 API Routes
# -------------------------------------------------------

@app.get("/health")
def health_check():
    """Simple endpoint to verify that API is alive."""
    return {"status": "ok"}


@app.post("/recommend")
def recommend(input: QueryInput):
    """Return top assessment recommendations based on query."""
    try:
        if not input.query.strip():
            return {"error": "Empty query provided"}

        results = recommender.recommend(input.query, top_k=10)
        formatted_results = []

        for r in results:
            formatted_results.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "test_type": [t.strip() for t in str(r.get("test_type", "")).split(",") if t.strip()]
            })

        return {"recommended_assessments": formatted_results}

    except Exception as e:
        print("❌ Error in recommend:", e)
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    """Serve the web frontend."""
    return FileResponse("index.html")
