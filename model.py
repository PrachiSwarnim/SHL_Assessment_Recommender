import numpy as np
import urllib.parse
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SHLRecommender:
    def __init__(self, catalog_df):
        """
        Initialize recommender with catalog DataFrame.
        Uses Assessment_Name as text column and includes Remote/Adaptive/Test Type metadata.
        """
        self.catalog_df = catalog_df.copy()

        # --- Validate and normalize columns ---
        required_cols = ["Assessment_Name", "Assessment_Url"]
        for col in required_cols:
            if col not in self.catalog_df.columns:
                raise ValueError(f"❌ Missing required column: {col}")

        # Normalize column names
        self.catalog_df.columns = [c.strip().replace(" ", "_").title() for c in self.catalog_df.columns]

        if "Test_Type" not in self.catalog_df.columns:
            self.catalog_df["Test_Type"] = "-"

        # --- Build TF-IDF Matrix ---
        self.text_col = "Assessment_Name"
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.catalog_df[self.text_col].fillna(""))

    # ------------------------------------------------------------
    # Infer SHL Test Type (A, B, C, D, E, K, P, S)
    # ------------------------------------------------------------
    def infer_test_type(self, text):
        text = text.lower()

        test_type_keywords = {
            "A": ["aptitude", "ability", "numerical", "verbal", "reasoning", "logic", "analytical"],
            "B": ["situational", "judgement", "judgment", "biodata", "scenario", "context"],
            "C": ["competency", "competencies", "skills profile", "behavioral competency"],
            "D": ["development", "360", "feedback", "growth", "coach", "learning"],
            "E": ["assessment", "exercise", "simulation", "case study", "task"],
            "K": ["python", "java", "sql", "excel", "technical", "knowledge", "skill", "coding",
                  "developer", "automata", "test", "data", "it", "software"],
            "P": ["personality", "behavior", "behaviour", "opq", "leadership", "communication",
                  "team", "interpersonal", "values", "emotional", "traits", "motivation"],
            "S": ["simulation", "roleplay", "virtual", "scenario-based"]
        }

        for key, words in test_type_keywords.items():
            if any(w in text for w in words):
                return key

        return "-"

    # ------------------------------------------------------------
    # Recommendation Logic
    # ------------------------------------------------------------
    def recommend(self, query: str, top_k: int = 10):
        if not query.strip():
            return []

        # Compute similarity
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        df = self.catalog_df.copy()
        df["score"] = np.round(scores, 4).astype(float)

        # Handle multi-type or missing Test_Type
        df["Test_Type"] = df.apply(
            lambda row: row["Test_Type"]
            if isinstance(row["Test_Type"], str) and any(t in row["Test_Type"].upper() for t in list("ABCDEKPS"))
            else self.infer_test_type(f"{row.get('Assessment_Url', '')} {row.get('Assessment_Name', '')}"),
            axis=1
        )

        # Sort by score
        df = df.sort_values("score", ascending=False)

        # Balance test types if possible
        k_tests = df[df["Test_Type"].str.contains("K", na=False)].head(top_k // 2)
        p_tests = df[df["Test_Type"].str.contains("P", na=False)].head(top_k // 2)
        remaining = df[~df["Test_Type"].str.contains("K|P", na=False)].head(top_k - len(k_tests) - len(p_tests))

        combined = pd.concat([k_tests, p_tests, remaining]).drop_duplicates(subset="Assessment_Url", keep="first")
        combined = combined.sort_values("score", ascending=False).head(top_k)

        # Format results for UI
        safe_data = []
        for _, row in combined.iterrows():
            url = str(row.get("Assessment_Url", "")).strip()
            decoded_url = urllib.parse.unquote(url)
            name_part = decoded_url.rstrip("/").split("/")[-1]
            display_name = re.sub(r"[-_]+", " ", name_part).strip().title()
            display_name = re.sub(r"\bNew\b", "– New", display_name)

            if not display_name or display_name.lower() == "nan":
                display_name = row.get("Assessment_Name", "SHL Assessment")

            safe_data.append({
                "name": display_name,
                "url": url,
                "test_type": row["Test_Type"],
                "remote_testing": row.get("Remote_Testing", "-"),
                "adaptive_irt": row.get("Adaptive_Irt", "-"),
                "score": float(row["score"])
            })

        return safe_data
