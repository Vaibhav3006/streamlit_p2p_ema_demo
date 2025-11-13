import boto3
import pandas as pd
import json
# 'time' is no longer needed as we aren't polling
# 'json' might be useful for printing, but not required by the functions

# ==============================================================================
# === 1. NEW FUNCTIONS FOR LOCAL FILE PROCESSING ===
# ==============================================================================
import boto3, os

def make_textract_client(region_name: str = "us-east-1", profile: str | None = None):
    """
    Creates a Textract client using an AWS CLI profile if provided,
    else falls back to env vars/default credentials.
    """
    if profile:
        session = boto3.Session(profile_name=profile, region_name=region_name)
        return session.client("textract")
    # Allow AWS_PROFILE env var to drive the profile too
    env_profile = os.getenv("AWS_PROFILE")
    if env_profile:
        session = boto3.Session(profile_name=env_profile, region_name=region_name)
        return session.client("textract")
    # Default: use whatever creds are configured as 'default'
    return boto3.client("textract", region_name=region_name)

def run_text_detection_local(file_path, region_name="us-east-1"):
    """
    Runs SYNCHRONOUS text detection on a local file.
    
    Args:
        file_path (str): The path to your local PDF or image file.
        region_name (str): The AWS region.

    Returns:
        dict: The parsed JSON object with the "extracted_text" key.
    """
    textract_client = make_textract_client(region_name=region_name, profile="personal-textract")
    
    print(f"Running text detection on local file: {file_path}")
    
    # 1. Read the file bytes
    try:
        with open(file_path, "rb") as document_file:
            file_bytes = document_file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return {"extracted_text": ""}

    # 2. Call the synchronous API
    # Note: Using detect_document_text
    response = textract_client.detect_document_text(
        Document={'Bytes': file_bytes}
    )
    
    print("Detection complete. Parsing text...")

    # 3. Parse the response
    # The 'parse_textract_text' function expects a list of 'pages' (responses).
    # The sync response is a single object, so we wrap it in a list.
    parsed_data = parse_textract_text([response])
    
    return parsed_data

def run_analysis_local(file_path, feature_types, region_name="us-east-1"):
    """
    Runs SYNCHRONOUS document analysis (Forms or Tables) on a local file.
    
    Args:
        file_path (str): The path to your local PDF or image file.
        feature_types (list): A list of strings, e.g., ["TABLES"], ["FORMS"],
                              or ["TABLES", "FORMS"].
        region_name (str): The AWS region.

    Returns:
        dict: The full response from textract_client.analyze_document
    """
    textract_client = make_textract_client(region_name=region_name, profile="personal-textract")
    
    print(f"Running analysis for {feature_types} on local file: {file_path}")
    
    # 1. Read the file bytes
    try:
        with open(file_path, "rb") as document_file:
            file_bytes = document_file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None # Return None on failure

    # 2. Call the synchronous API
    # Note: Using analyze_document
    response = textract_client.analyze_document(
        Document={'Bytes': file_bytes},
        FeatureTypes=feature_types
    )
    
    print("Analysis complete.")
    return response


def run_table_analysis_local(file_path, region_name="us-east-1"):
    """
    1) Runs sync Document Analysis with TABLES on a local file
    2) Retrieves & parses them into DataFrames
    """
    # Step 1: Run the synchronous analysis
    response = run_analysis_local(
        file_path=file_path,
        feature_types=["TABLES"],
        region_name=region_name
    )
    
    if response is None:
        return [] # Return empty list if analysis failed
        
    print("Parsing table data...")

    # Step 2: Get all blocks (sync response has all blocks in one go)
    all_blocks = response.get("Blocks", [])

    # Step 3: Parse to list-of-lists
    tables = parse_table_data(all_blocks)
    
    # Step 4: Convert each table to a pandas DataFrame
    dataframes = tables_to_dataframes(tables)

    return dataframes

def run_form_analysis_local(file_path, region_name="us-east-1"):
    """
    1) Runs sync Document Analysis with FORMS on a local file
    2) Retrieves & parses them into a key-value dict
    3) Returns a JSON object (Python dict) with cleaned key-value pairs
    """
    
    # Step 1: Run the synchronous analysis
    response = run_analysis_local(
        file_path=file_path,
        feature_types=["FORMS"], # or ["TABLES", "FORMS"]
        region_name=region_name
    )
    
    if response is None:
        return {} # Return empty dict if analysis failed

    print("Parsing form data...")

    # Step 2: Get all blocks
    all_blocks = response.get("Blocks", [])

    # Step 3: Parse forms to get key-value pairs
    kv_pairs = parse_form_data(all_blocks)
    
    # Step 4: Clean the dictionary
    cleaned_dict = {}
    for key, value in kv_pairs.items():
        if not value.strip(): # Skip entries if value is empty
            continue
        
        new_key = key.rstrip(":") # Remove trailing colon
        cleaned_dict[new_key] = value.strip()
    
    return cleaned_dict


# ==============================================================================
# === 2. REUSABLE PARSING LOGIC (MOSTLY UNCHANGED) ===
# ==============================================================================

def parse_textract_text(pages):
    """
    Gathers all text lines into one string, then returns JSON with
    a single key "extracted_text".
    
    (This function works as-is, since we pass it [response])
    """
    all_lines = []
    for page_data in pages:
        for block in page_data.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                text_line = block.get("Text", "")
                all_lines.append(text_line)

    combined_text = "\n".join(all_lines)
    result = {
        "extracted_text": combined_text
    }
    return result

def extract_cell_text(cell_block, block_map):
    """
    Gather text (from WORD children) inside a CELL block.
    """
    text = []
    if "Relationships" in cell_block:
        for rel in cell_block["Relationships"]:
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    word_block = block_map.get(child_id)
                    if word_block and word_block["BlockType"] == "WORD":
                        text.append(word_block["Text"])
    return " ".join(text)

def parse_table_data(all_blocks):
    """
    Returns a list of tables, where each table is a list-of-lists (rows->columns).
    (This function works as-is)
    """
    block_map = {block["Id"]: block for block in all_blocks}
    tables = []
    
    for block in all_blocks:
        if block["BlockType"] == "TABLE":
            table_rows = {}
            if "Relationships" in block:
                for rel in block["Relationships"]:
                    if rel["Type"] == "CHILD":
                        for cell_id in rel["Ids"]:
                            cell_block = block_map.get(cell_id)
                            if cell_block and cell_block["BlockType"] == "CELL":
                                row_idx = cell_block["RowIndex"]
                                col_idx = cell_block["ColumnIndex"]
                                cell_text = extract_cell_text(cell_block, block_map)
                                
                                if row_idx not in table_rows:
                                    table_rows[row_idx] = {}
                                table_rows[row_idx][col_idx] = cell_text
            
            # Convert row dictionary to a list-of-lists
            table_data = []
            for row_index in sorted(table_rows.keys()):
                cols_in_row = table_rows[row_index]
                row_content = [cols_in_row[col_index] for col_index in sorted(cols_in_row.keys())]
                table_data.append(row_content)
            
            tables.append(table_data)
    
    return tables

def tables_to_dataframes(tables):
    """
    Given a list of tables (list-of-lists), convert each to a pandas DataFrame.
    (This function works as-is)
    """
    dataframes = []
    for table in tables:
        if not table:
            continue

        max_cols = max(len(row) for row in table)
        
        # Handle case where table might be jagged, pad with None
        padded_table = []
        for row in table:
            padded_row = row + [None] * (max_cols - len(row))
            padded_table.append(padded_row)
            
        # Try to use first row as header, but fall back if it's empty
        header = padded_table[0]
        if all(col is None or col == '' for col in header):
            # First row is empty, use generic names
            col_names = [f"Column{i+1}" for i in range(max_cols)]
            df = pd.DataFrame(padded_table, columns=col_names)
        else:
            # Use first row as header
            df = pd.DataFrame(padded_table[1:], columns=header)
            
        dataframes.append(df)

    return dataframes

def dataframes_to_json(dataframes):
    """
    Converts a list of pandas DataFrames into a JSON object (Python dictionary).
    (This function works as-is)
    """
    result = {}
    for idx, df in enumerate(dataframes, start=1):
        header = list(df.columns)
        rows = df.values.tolist()
        result[f"DataFrame_{idx}"] = {
            "columns": header,
            "rows": rows
        }
    return result

def parse_form_data(blocks):
    """
    Extract key-value pairs from KEY_VALUE_SET blocks.
    (This function works as-is)
    """
    block_map = {block["Id"]: block for block in blocks}
    
    key_map = {}
    value_map = {}

    for block in blocks:
        if block["BlockType"] == "KEY_VALUE_SET":
            if "KEY" in block.get("EntityTypes", []):
                key_map[block["Id"]] = block
            else:
                value_map[block["Id"]] = block

    kv_pairs = {}
    for key_id, key_block in key_map.items():
        key_text = extract_text(key_block, block_map)
        value_block_id = find_value_block_id(key_block)
        
        if value_block_id in value_map:
            value_text = extract_text(value_map[value_block_id], block_map)
        else:
            value_text = ""
        kv_pairs[key_text] = value_text

    return kv_pairs

def extract_text(block, block_map):
    """
    Extract text from WORD or SELECTION_ELEMENT children.
    (This function works as-is, with a small safety check)
    """
    text = []
    if "Relationships" in block:
        for rel in block["Relationships"]:
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    child_block = block_map.get(child_id)
                    if not child_block:
                        continue
                        
                    if child_block["BlockType"] == "WORD":
                        text.append(child_block["Text"])
                    elif child_block["BlockType"] == "SELECTION_ELEMENT":
                        if child_block["SelectionStatus"] == "SELECTED":
                            text.append("SELECTED")
    return " ".join(text)

def find_value_block_id(key_block):
    """
    Find the ID of the VALUE block associated with a KEY block.
    (This function works as-is)
    """
    if "Relationships" in key_block:
        for rel in key_block["Relationships"]:
            if rel["Type"] == "VALUE":
                # A key can only have one value
                if rel["Ids"]:
                    return rel["Ids"][0]
    return None

# single_file_processors.py
from pathlib import Path
import csv
import json

# Assumes these exist in your environment:
#   run_form_analysis_local(file_path, region_name)
#   run_table_analysis_local(file_path, region_name)
#   run_text_detection_local(file_path, region_name)

def _df_to_obj(df):
    cols = list(df.columns)
    rows = df.astype(object).where(df.notnull(), None).values.tolist()
    return {"columns": cols, "rows": rows}

def process_pdf(
    file_path: str,
    region_name: str = "us-east-1",
    include_text: bool = True,
    table_mode: str = "first",  # "first" | "concat"
):
    """
    Returns a simple JSON for a single PDF:
    {"forms": <dict>, "text": <str>, "table": {"columns":[...], "rows":[...]}|None}
    """
    # Import here to avoid circulars if you place this in same package
    # from your_module import (  # <-- replace with actual module where these live
    #     run_form_analysis_local,
    #     run_table_analysis_local,
    #     run_text_detection_local,
    # )

    forms = run_form_analysis_local(file_path, region_name) or {}

    # Tables
    table_obj = None
    dfs = run_table_analysis_local(file_path, region_name) or []
    if dfs:
        if table_mode == "concat" and len(dfs) > 1:
            import pandas as pd
            norm = []
            for d in dfs:
                dd = d.copy()
                dd.columns = [str(c) for c in dd.columns]
                norm.append(dd)
            all_cols = sorted({c for d in norm for c in d.columns})
            norm = [d.reindex(columns=all_cols) for d in norm]
            concat = pd.concat(norm, ignore_index=True)
            table_obj = _df_to_obj(concat)
        else:
            table_obj = _df_to_obj(dfs[0])

    # Text
    text = ""
    if include_text:
        t = run_text_detection_local(file_path, region_name) or {}
        text = t.get("extracted_text", "")

    return {"forms": forms, "text": text, "table": table_obj}

def process_grn_csv(
    file_path: str,
    encoding: str = "utf-8",
    limit_rows: int | None = None,
):
    """
    Returns a simple JSON for a single GRN CSV:
    {"rows": [ {col: val, ...}, ... ]}
    """
    rows = []
    with open(file_path, "r", encoding=encoding, newline="") as fin:
        reader = csv.DictReader(fin)
        for i, row in enumerate(reader, 1):
            rows.append(row)
            if limit_rows and i >= limit_rows:
                break
    return {"rows": rows}

def process_one(
    file_path: str,
    region_name: str = "us-east-1",
    include_text: bool = True,
    table_mode: str = "first",
    encoding: str = "utf-8",
    limit_rows: int | None = None,
):
    """
    Convenience router: chooses PDF vs CSV automatically.
    """
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return process_pdf(
            file_path=file_path,
            region_name=region_name,
            include_text=include_text,
            table_mode=table_mode,
        )
    elif suffix == ".csv":
        return process_grn_csv(
            file_path=file_path,
            encoding=encoding,
            limit_rows=limit_rows,
        )
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
