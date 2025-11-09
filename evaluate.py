import pandas as pd
from model import SHLRecommender


def recall_at_k(predicted, relevant, k=10):
    """Compute recall@k for one query."""
    if not relevant:
        return 0
    retrieved = predicted[:k]
    hits = len(set(retrieved) & set(relevant))
    return hits / len(relevant)


# --- Load workbook and specific sheets explicitly ---
workbook_path = "Gen_AI Dataset.xlsx"

try:
    train_df = pd.read_excel(workbook_path, sheet_name="Train-Set")
    print(f"✅ Loaded 'Train-Set' sheet with {len(train_df)} rows.")
except Exception as e:
    raise ValueError(f"❌ Failed to read 'Train-Set' sheet: {e}")

try:
    test_df = pd.read_excel(workbook_path, sheet_name="Test-Set")
    print(f"✅ Loaded 'Test-Set' sheet with {len(test_df)} rows.")
except Exception:
    test_df = None
    print("⚠️ No 'Test-Set' sheet found. Proceeding with Train-Set only.")

# --- Build catalog from both sheets ---
if test_df is not None:
    combined_catalog = pd.concat([train_df, test_df], ignore_index=True)
else:
    combined_catalog = train_df.copy()

# Drop duplicates based on the Assessment_url column
if "Assessment_url" in combined_catalog.columns:
    combined_catalog = combined_catalog.drop_duplicates(subset="Assessment_url")
else:
    raise ValueError("❌ Expected column 'Assessment_url' not found in dataset.")

print(f"✅ Constructed catalog with {len(combined_catalog)} unique assessments.")

# --- Initialize recommender ---
recommender = SHLRecommender(combined_catalog)

# --- Define column names ---
query_col = "Query"
url_col = "Assessment_url"

recalls = []
print("\n🧠 Evaluating Recall@10 on 'Train-Set'\n")

for i, row in train_df.iterrows():
    query = str(row[query_col]).strip()
    if not query:
        continue

    relevant_urls = [str(row[url_col]).strip().lower()]
    preds = recommender.recommend(query, top_k=10)

    # Normalize predicted URLs for comparison
    predicted_urls = [
        str(p.get("Assessment_url", p.get("url", ""))).strip().lower()
        for p in preds
    ]

    rec = recall_at_k(predicted_urls, relevant_urls, k=10)
    recalls.append(rec)
    print(f"🔹 {i+1}. {query[:70]}... → Recall@10 = {rec:.2f}")

mean_recall = sum(recalls) / len(recalls) if recalls else 0
print(f"\n✅ Mean Recall@10 = {mean_recall:.3f}")

# Save metric for API/UI display
with open("metrics.txt", "w") as f:
    f.write(str(round(mean_recall, 3)))
