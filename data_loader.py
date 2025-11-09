import pandas as pd

def load_shl_data(filepath):
    """
    Loads and normalizes the SHL dataset from Excel.
    It auto-detects the catalog sheet (containing assessments)
    and renames its columns for the recommender model.
    """
    xls = pd.ExcelFile(filepath)
    sheets = xls.sheet_names
    data = {}

    for sheet in sheets:
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Check if this sheet is the catalog (contains assessment info)
        if any(k in df.columns for k in ["assessment_name", "assessment_url", "assessment_description"]):
            df = df.rename(columns={
                "assessment_name": "name",
                "assessment_url": "url",
                "assessment_description": "description",
                "test_type": "test_type"
            })
            data["catalog"] = df
        else:
            data[sheet] = df

    # fallback in case no 'catalog' key found
    if "catalog" not in data and data:
        data["catalog"] = list(data.values())[0]

    return data
