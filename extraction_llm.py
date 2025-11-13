from openai import OpenAI
import os
import json
import openai
import streamlit as st

api_key = st.secrets["OPENAI_API_KEY"]

def extract_receipt_details(receipt_data_str):
    
    
    client = OpenAI(api_key=api_key)

    prompt = """
System / Role Instructions
You are a document extraction specialist. Extract and structure information from the provided document data into a clean JSON format.

INPUT DATA:
{input_json}

EXTRACTION REQUIREMENTS:
1. Identify the document type: "PO" (Purchase Order), "GRN" (Goods Receipt Note), or "Invoice"
2. Extract the PO Number, Invoice Number and GRN
3. Extract the Date (keep original format)
4. Extract the Vendor name
5. Extract the Currency code
6. Parse all line items into a structured items array
7. Extract the total amount

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure:
{
  "document_type": "PO" | "GRN" | "Invoice",
  "po_number": "string",
  "inv_number": "string",
  "grn_number": "string",
  "date": "string",
  "vendor_name": "string",
  "currency": "string",
  "items": [
    {
      "sku_id": "string or null if not available",
      "name": "string",
      "quantity": number,
      "unit_price": number,
      "line_total": number
    }
  ],
  "total_amount": number
}

PARSING RULES:
- Remove currency symbols from numeric values and convert to float
- Use commas or periods appropriately based on European (1.234,56) or US (1,234.56) formats
- If SKU ID is not present in the data, use null
- Extract item names, quantities, and prices from the table_output_json array
- Document type should be inferred from the extracted_text (look for "Purchase order", "Invoice", or "GRN")
- Clean any typos like "rotal" → "total"
- keep  "po_number", "inv_number" and grn_number" as "not sepcified" if not present n the document
- SKU id can only be like "SKU1001250" format. An SKU ID cannot end with a letter and can only contain "SKU" letter in this order only.

Return only the JSON output, no explanations.
"""

    combined_prompt = prompt + "\n\nHere is the input data JSON:\n" + receipt_data_str

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": combined_prompt}],
        }],
    )
    
    content_string = response.choices[0].message.content
    
    if content_string:
        try:
            # If content IS a valid JSON string:
            output = json.loads(content_string)
        except json.JSONDecodeError:
            # Else if it's just a regular string:
            output = content_string
    else:
    # Else (if there was no content to begin with):
        output = "Error: Response content is missing or empty."


    return output
