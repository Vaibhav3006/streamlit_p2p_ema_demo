FX_RATE_FILE = "/Users/sakshisakshi/Desktop/EmaAssignment/streamlit_demo/Data/fx_rates_sample.csv"
INPUT_DIRS = [
    "/Users/sakshisakshi/Desktop/EmaAssignment/streamlit_demo/Data/Incoming",    # PDFs (PO/Invoices)
    "/Users/sakshisakshi/Desktop/EmaAssignment/streamlit_demo/Data/Incoming",   # CSVs (GRNs)
]
OUTPUT_DIR = "/Users/sakshisakshi/Desktop/EmaAssignment/streamlit_demo/Data/Output"       # where final CSVs will be appended
AWS_REGION = "us-east-1"
WORKERS = 8                                # tune 6–12 based on your machine
BATCH_SIZE = 32                            # files per batch before appending