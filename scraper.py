# Importing libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

# Base URL for SHL individual assessment listings
BASE_URL = "https://www.shl.com/products/product-catalog/?start={}&type=1&type=1"

# Output file for the scraped data
OUTPUT_FILE = "SHL_Scraped_Assessments.csv"


def clean_text(text):
    """Cleans up any extra spaces, tabs, or newlines from the extracted text."""
    return re.sub(r"\s+", " ", text.strip()) if text else ""


def parse_table_row(row):
    """Extracts key assessment details from a single HTML <tr> row."""
    cols = row.find_all("td")
    if len(cols) < 4:
        return None  # Skip incomplete or malformed rows

    # Extracting assessment name and URL
    name_tag = cols[0].find("a")
    if not name_tag:
        return None

    name = clean_text(name_tag.text)
    url = "https://www.shl.com" + name_tag["href"].strip()

    # Skipping job bundles, packages, or pre-packed solutions
    if any(kw in url.lower() for kw in ["solution", "bundle", "package", "suite"]):
        return None

    # Extracting Remote Testing and Adaptive/IRT availability
    remote = "Yes" if cols[1].find("span", class_="-yes") else "No"
    adaptive = "Yes" if cols[2].find("span", class_="-yes") else "No"

    # Extracting test type tags (A, B, C, D, E, K, P, S)
    types = [t.text.strip() for t in cols[3].find_all("span", class_="product-catalogue__key")]
    test_type = ", ".join(types) if types else "-"

    return {
        "Assessment_Name": name,
        "Assessment_Url": url,
        "Remote_Testing": remote,
        "Adaptive_IRT": adaptive,
        "Test_Type": test_type
    }


def main():
    """Scrapes all individual SHL assessments across all catalog pages."""
    all_rows = []
    total_collected = 0

    # There are 34 pages of results, each showing 12 items
    start_values = [i * 12 for i in range(34)]

    for start in start_values:
        page_url = BASE_URL.format(start)
        print(f"📄 Fetching page starting at {start} → {page_url}")

        try:
            resp = requests.get(page_url, timeout=20)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")

            # Skipping pages with empty results
            if not table:
                continue

            # Parsing each table row excluding the header
            for row in table.find_all("tr")[1:]:
                data = parse_table_row(row)
                if data:
                    all_rows.append(data)
                    total_collected += 1

            time.sleep(1.5)  

        except Exception:
            time.sleep(3)
            continue

    # Convert all scraped rows into a DataFrame and remove duplicates
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["Assessment_Url"])

    # Save final dataset
    df.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
