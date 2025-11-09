# SHL Assessment Recommender

The **SHL Assessment Recommender** is a web application that suggests the most relevant SHL assessments based on a job description or query.  
It uses **TF-IDF text similarity** to match job-related keywords with the SHL assessment catalog. It organizes results by **test type (A–S)** and displays them in a user-friendly interface.

---

## Project Overview

This project automates finding SHL assessments for recruitment and talent evaluation.  
Given a job description or requirement, it recommends suitable SHL tests such as **Aptitude (A)**, **Personality (P)**, or **Knowledge-based (K)** assessments.

### Key Features
- Intelligent recommendation engine using TF-IDF and cosine similarity.
- SHL catalog with over 400 individual test solutions.
- Automatic inference of test types (A–S categories like Ability, Personality, Knowledge, etc.).
- FastAPI backend with a responsive HTML/JavaScript frontend.
- Live deployment on Render.

---

## System Architecture

The project follows a **modular 6-layer pipeline** architecture, from data collection to frontend interaction:

### 1. Data Layer (Catalog Source)
- SHL’s **product catalog** was scraped using BeautifulSoup and Requests.  
- Each record includes:
  - Assessment Name  
  - Assessment URL  
  - Remote Testing availability  
  - Adaptive/IRT support  
  - Test Type (A–S codes)  
- The final dataset is stored as `SHL_Scraped_Assessments.csv`.

### 2. Model Layer (Recommender Engine)
- The recommender (`model.py`) uses **TF-IDF vectorization** and **cosine similarity** to measure the closeness between job descriptions and assessment titles.  
- It infers missing test types using keyword-based classification (A–S labels).  
- The top 10 most relevant assessments are ranked and returned.

### 3. Backend Layer (FastAPI Service)
- The **FastAPI backend** (`main.py`) powers the API.  
- Endpoints:
  - `POST /recommend` → Returns recommended assessments.  
  - `GET /health` → Confirms API uptime.  
- The backend uses CORS to allow communication with the frontend.  
- Results are returned as structured JSON objects.

### 4. Frontend Layer (User Interface)
- The **HTML/JavaScript UI** (`index.html`) lets users input job descriptions.  
- When submitted, it sends the query to the backend and displays results dynamically.  
- Each recommendation includes:
  - Assessment Name  
  - Clickable Assessment URL  
  - **Color-coded Test Type Tags** with hover tooltips showing full category names.

### 5. Evaluation Layer (Model Validation)
- Model performance is validated using the **Recall@10** metric.  
- The evaluation script (`evaluate_model.py`) checks how often the correct assessment appears in the top 10 predictions.  
- The mean recall value is saved in `metrics.txt` for tracking performance.

### 6. Deployment Layer (Render Cloud)
- The system is deployed on **Render**.  
- FastAPI serves the API endpoints while the frontend is hosted as a static HTML file.  
- The deployment ensures scalability; adding new assessments only needs updating the CSV file.

---

## Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend** | HTML5, CSS3, JavaScript |
| **Backend** | FastAPI |
| **Model** | TF-IDF (Scikit-learn), Cosine Similarity |
| **Scraper** | BeautifulSoup4, Requests |
| **Deployment** | Render |
| **Language** | Python 3.10+ |

---

## Evaluation Metric

The model’s performance was evaluated using **Mean Recall@10**.

| Metric | Description |
|---------|--------------|
| **Recall@10** | Fraction of relevant assessments retrieved among the top 10. |
| **Mean Recall@10** | Average recall across all test queries. |

Example result: Mean Recall@10 = 0.87

---

## Model Workflow

1. Cleaned and combined SHL catalog data.
2. Built a TF-IDF vectorizer on assessment names.
3. Computed cosine similarity with incoming queries.
4. Classified tests into A to S categories.
5. Ranked the top 10 recommendations and served them via API.

---

## Running Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/shl-assessment-recommender.git
cd shl-assessment-recommender
```

### 2. Create a Virtual Environment
```bash
python -m venv shl
source shl/bin/activate  # (Linux/macOS)
shl\Scripts\activate     # (Windows)
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Start the FastAPI Server
Open ```index.html``` in the browser.

## Deployment

The live project is deployed on Render:

Live Demo: https://shl-assessment-recommender-ellx.onrender.com

## Optimization Journey

The model was improved through several rounds of changes to boost recommendation accuracy and variety.  
Here’s a summary of each phase of optimization and its effect on **Mean Recall@10**.

| **Phase** | **Improvement** | **Mean Recall@10** |
|-----------|------------------|---------------------|
| **Initial TF-IDF model** | Used basic text similarity with TF-IDF and cosine similarity | 0.64 |
| **Catalog cleanup** | Eliminated pre-packaged test bundles and standardized individual assessments | 0.75 |
| **Test type inference** | Added smart A-S categorization with keyword-based classification | 0.83 |
| **Balanced output** | Introduced mixed-type ranking (A, K, P) to improve recommendation variety | **0.87** |

---

**Final Performance:**  
The optimized system achieved a **Mean Recall@10 of 0.87**. This shows a strong connection between job descriptions and SHL assessment recommendations.
## Author

Prachi Swarnim
Data Science & Analytics Enthusiast
Email: prachi.swarnim@gmail.com
LinkedIn: www.linkedin.com/in/prachi-swarnim29