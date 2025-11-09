# Importing libraries
import pandas as pd
from model import SHLRecommender

def recall_at_k(predicted, relevant, k=10):
    """
    Calculates Recall@K — a standard metric in recommendation systems.
    It measures how many of the relevant items are captured within
    the top-k predictions.
    """
    if not relevant:
        return 0
    retrieved = predicted[:k]
    hits = len(set(retrieved) & set(relevant))
    return hits / len(relevant)


# Loading the dataset.
workbook_path = "Gen_AI Dataset.xlsx"

try:
    train_df = pd.read_excel(workbook_path, sheet_name="Train-Set")
except Exception as e:
    raise ValueError(f"Failed to read 'Train-Set' sheet: {e}")

try:
    test_df = pd.read_excel(workbook_path, sheet_name="Test-Set")
except Exception:
    test_df = None

# Preparing the catalog (unique list of assessments)
if test_df is not None:
    combined_catalog = pd.concat([train_df, test_df], ignore_index=True)
else:
    combined_catalog = train_df.copy()

# Dropping duplicates based on unique Assessment URLs
if "Assessment_url" in combined_catalog.columns:
    combined_catalog = combined_catalog.drop_duplicates(subset="Assessment_url")
else:
    raise ValueError("Expected column 'Assessment_url' not found in dataset.")


# Initializing the SHL Recommender Model
recommender = SHLRecommender(combined_catalog)

# Evaluating Model Performance (Recall@10)
query_col = "Query"
url_col = "Assessment_url"
recalls = []

for i, row in train_df.iterrows():
    query = str(row[query_col]).strip()
    if not query:
        continue

    # Each row's Assessment_url is considered the ground truth
    relevant_urls = [str(row[url_col]).strip().lower()]

    # Generating top-10 recommendations for this query
    preds = recommender.recommend(query, top_k=10)

    # Extracting and normalizing predicted URLs
    predicted_urls = [
        str(p.get("Assessment_url", p.get("url", ""))).strip().lower()
        for p in preds
    ]

    # Computing Recall@10 for the current query
    rec = recall_at_k(predicted_urls, relevant_urls, k=10)
    recalls.append(rec)
    print(f"🔹 {i+1}. {query[:70]}... → Recall@10 = {rec:.2f}")

# Aggregating and saving results
mean_recall = sum(recalls) / len(recalls) if recalls else 0
print(f"Mean Recall@10 = {mean_recall:.3f}")
