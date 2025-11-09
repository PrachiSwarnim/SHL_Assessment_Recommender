import pandas as pd
from model import SHLRecommender

# --- Load Data ---
workbook_path = "Gen_AI Dataset.xlsx"
catalog_path = "SHL_Scraped_Assessments.csv"

# --- Read Test Queries ---
test_df = pd.read_excel(workbook_path, sheet_name="Test-Set", dtype=str)
print(f"✅ Loaded 'Test-Set' with {len(test_df)} queries.")
print(f"🧾 Example Query:\n{test_df['Query'].iloc[0][:250]}...")

# --- Read SHL Catalog ---
catalog_df = pd.read_csv(catalog_path)
print(f"✅ Loaded SHL catalog from '{catalog_path}' with {len(catalog_df)} assessments.")

# --- Validate Columns ---
required_cols = ["Assessment_Name", "Assessment_Url"]
missing_cols = [col for col in required_cols if col not in catalog_df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns in catalog CSV: {missing_cols}")

# --- Initialize Recommender ---
print("\n⚙️ Initializing SHL Recommender using scraped catalog...")
recommender = SHLRecommender(catalog_df)
print("✅ Model initialized successfully.\n")

# --- Generate Recommendations ---
rows = []
query_col = "Query"

print("🧠 Generating recommendations for Test-Set...\n")

for i, row in test_df.iterrows():
    query = str(row.get(query_col, "")).strip()
    if not query:
        print(f"⚠️ Skipping empty query at row {i}")
        continue

    try:
        preds = recommender.recommend(query, top_k=10)
        for p in preds:
            rows.append({
                "Query": query,
                "Assessment_Url": p.get("url", p.get("Assessment_Url", ""))
            })
        print(f"🔹 Processed {i+1}/{len(test_df)} → {len(preds)} recommendations generated.")
    except Exception as e:
        print(f"⚠️ Error processing query {i+1}: {e}")

# --- Create DataFrame and Save as-is ---
submission_df = pd.DataFrame(rows, columns=["Query", "Assessment_Url"])
submission_df.to_csv("submission.csv", index=False, encoding="utf-8-sig")

print(f"\n✅ Submission file created successfully: 'submission.csv' ({len(submission_df)} rows)")
print("📄 Columns: Query | Assessment_Url")
