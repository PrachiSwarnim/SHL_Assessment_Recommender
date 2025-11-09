# Importing libraries
import numpy as np
import urllib.parse
import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SHLRecommender:
    def __init__(self, catalog_df):
        """
        Initialize the recommender system with a given SHL assessment catalog.
        The catalog is a DataFrame containing assessment details such as name, URL, and test type.
        """
        self.catalog_df = catalog_df.copy()

        # Checking for essential columns
        required_cols = ["Assessment_Name", "Assessment_Url"]
        for col in required_cols:
            if col not in self.catalog_df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Cleaning up column names (capitalization and removing spaces)
        self.catalog_df.columns = [c.strip().replace(" ", "_").title() for c in self.catalog_df.columns]

        # Adding a default Test_Type column if not present
        if "Test_Type" not in self.catalog_df.columns:
            self.catalog_df["Test_Type"] = "-"

        # Preparing the TF-IDF matrix using Assessment_Name as the main textual feature
        self.text_col = "Assessment_Name"
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.catalog_df[self.text_col].fillna(""))

    # Infers SHL Test Type if missing or unclear
    def infer_test_type(self, text):
        """
        Try to guess the SHL test type (A, B, C, D, E, K, P, S)
        based on keywords found in the assessment name or URL.
        """
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

        # If a keyword matches, return the associated test type letter
        for key, words in test_type_keywords.items():
            if any(w in text for w in words):
                return key

        return "-"

    # Main Recommendation Logic
    def recommend(self, query: str, top_k: int = 10):
        """
        Given a query (like a job description or hiring requirement),
        this function returns the top_k most relevant SHL assessments.
        """
        if not query.strip():
            return []

        # Converting the query into TF-IDF vector form and calculating cosine similarity
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        # Cleaning up any NaN or infinity values
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Attaching similarity scores to the catalog
        df = self.catalog_df.copy()
        df["score"] = np.round(scores, 4).astype(float)

        # Ensuring each test has a valid Test_Type — infer if missing or unclear
        df["Test_Type"] = df.apply(
            lambda row: row["Test_Type"]
            if isinstance(row["Test_Type"], str) and any(t in row["Test_Type"].upper() for t in list("ABCDEKPS"))
            else self.infer_test_type(f"{row.get('Assessment_Url', '')} {row.get('Assessment_Name', '')}"),
            axis=1
        )

        # Ranking the assessments by descending similarity score
        df = df.sort_values("score", ascending=False)

        # Slightly balance between Knowledge (K) and Personality (P) tests for variety
        k_tests = df[df["Test_Type"].str.contains("K", na=False)].head(top_k // 2)
        p_tests = df[df["Test_Type"].str.contains("P", na=False)].head(top_k // 2)
        remaining = df[~df["Test_Type"].str.contains("K|P", na=False)].head(top_k - len(k_tests) - len(p_tests))

        combined = pd.concat([k_tests, p_tests, remaining]).drop_duplicates(subset="Assessment_Url", keep="first")
        combined = combined.sort_values("score", ascending=False).head(top_k)

        # Formatting final results for the frontend
        safe_data = []
        for _, row in combined.iterrows():
            url = str(row.get("Assessment_Url", "")).strip()
            decoded_url = urllib.parse.unquote(url)

            # Extracting a readable name from the URL
            name_part = decoded_url.rstrip("/").split("/")[-1]
            display_name = re.sub(r"[-_]+", " ", name_part).strip().title()
            display_name = re.sub(r"\bNew\b", "– New", display_name)

            # Fallback to original name if URL-derived name looks invalid
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
