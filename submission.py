# Importing Libraries
import pandas as pd
from model import SHLRecommender

# Loading Data
workbook_path = "Gen_AI Dataset.xlsx"
catalog_path = "SHL_Scraped_Assessments.csv"

# Reading Test Queries
test_df = pd.read_excel(workbook_path, sheet_name="Test-Set", dtype=str)

# Reading SHL Product Catalog
catalog_df = pd.read_csv(catalog_path)

# Validating Columns
required_cols = ["Assessment_Name", "Assessment_Url"]
missing_cols = [col for col in required_cols if col not in catalog_df.columns]

# Initializing Recommender
recommender = SHLRecommender(catalog_df)

# Generating Recommendations
rows = []
query_col = "Query"

for i, row in test_df.iterrows():
    query = str(row.get(query_col, "")).strip()
    if not query:
        continue

    try:
        preds = recommender.recommend(query, top_k=10)
        for p in preds:
            rows.append({
                "Query": query,
                "Assessment_Url": p.get("url", p.get("Assessment_Url", ""))
            })
    except Exception as e:
        print(f"Error processing query {i+1}: {e}")

# Creating the Pandas DataFrame and saving it into a CSV File
submission_df = pd.DataFrame(rows, columns=["Query", "Assessment_Url"])
submission_df.to_csv("submission.csv", index=False, encoding="utf-8-sig")