import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

BASE_URL = "https://www.shl.com/products/product-catalog/?start={}&type=1&type=1"
OUTPUT_FILE = "shl_individual_tests_full.csv"

def clean_text(text):
    """Normalize whitespace."""
    return re.sub(r"\s+", " ", text.strip()) if text else ""

def parse_table_row(row):
    """Extract test data from a single <tr>."""
    cols = row.find_all("td")
    if len(cols) < 4:
        return None

    # Name and URL
    name_tag = cols[0].find("a")
    if not name_tag:
        return None

    name = clean_text(name_tag.text)
    url = "https://www.shl.com" + name_tag["href"].strip()

    # Skip pre-packaged job bundles
    if any(kw in url.lower() for kw in ["solution", "bundle", "package", "suite"]):
        return None

    # Remote Testing / Adaptive
    remote = "Yes" if cols[1].find("span", class_="-yes") else "No"
    adaptive = "Yes" if cols[2].find("span", class_="-yes") else "No"

    # Test Type
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
    print("🚀 Scraping SHL Individual Test Solutions with Test Type info...\n")

    all_rows = []
    total_collected = 0
    start_values = [i * 12 for i in range(34)]  # 12 rows per page, ~34 pages

    for start in start_values:
        page_url = BASE_URL.format(start)
        print(f"📄 Fetching page: {page_url}")

        resp = requests.get(page_url, timeout=20)
        if resp.status_code != 200:
            print(f"⚠️ Skipped page start={start} (status {resp.status_code})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")

        if not table:
            print("⚠️ No table found, skipping...")
            continue

        for row in table.find_all("tr")[1:]:  # skip header row
            data = parse_table_row(row)
            if data:
                all_rows.append(data)
                total_collected += 1
                print(f"✅ {data['Assessment_Name']} | Type: {data['Test_Type']} | Remote: {data['Remote_Testing']}")

        print(f"➡️ Page with start={start} done. Total so far: {total_collected}")
        time.sleep(1.5)

    # Save all results
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["Assessment_Url"])
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n🎯 Done!")
    print(f"✅ Total tests collected: {len(df)}")
    print(f"📁 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
