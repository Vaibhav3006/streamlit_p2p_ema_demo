import pandas as pd
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd

def _dedupe_header(df: pd.DataFrame, id_col: str, date_col: str = "doc_date", fx_col: str = "fx_rate_doc_to_usd") -> pd.DataFrame:
    """
    Keep exactly one header row per document id (PO/Invoice).
    Preference order:
      1) latest doc_date
      2) row WITH FX populated over row without FX
    Safe if date/fx columns are missing.
    """
    if not isinstance(df, pd.DataFrame) or df.empty or id_col not in df.columns:
        return df

    out = df.copy()

    # parse date if present
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    # mark fx nulls if present
    if fx_col in out.columns:
        out["_fx_isnull"] = out[fx_col].isna()
    else:
        out["_fx_isnull"] = False

    # stable sort so the "best" row ends up last within each id
    sort_cols = [id_col]
    if date_col in out.columns:
        sort_cols.append(date_col)
    sort_cols.append("_fx_isnull")
    out = out.sort_values(sort_cols, na_position="last")

    # keep the last row per id
    out = (
        out.drop(columns="_fx_isnull", errors="ignore")
           .drop_duplicates(subset=[id_col], keep="last")
           .reset_index(drop=True)
    )
    return out


class DocumentProcessor:
    """
    Processes normalized JSON documents and sorts them into five
    separate DataFrames for reconciliation:
    1. PO_Aggregate (Header-level POs)
    2. PO_LineItem (Line-level POs)
    3. INV_Aggregate (Header-level Invoices)
    4. INV_LineItem (Line-level Invoices)
    5. GRN_LineItem (Line-level GNRs)
    """
   
    def __init__(self):
        """
        Initializes the schemas and row collectors for the five tables.
        """
        # --- 1. Define Column Schemas ---
        self.PO_AGG_COLS = [
            'po_number', 'doc_date', 'vendor_id', 'vendor_name', 'currency',
            'base_currency', 'fx_rate_doc_to_usd', 'total_amount',
            'total_amount_usd', 'exceptions'
        ]
       
        self.PO_LINE_COLS = [
            'po_number', 'line_number', 'sku_id', 'description',
            'quantity', 'unit_price', 'line_total',
            'unit_price_usd', 'line_total_usd'
        ]
       
        self.INV_AGG_COLS = [
            'invoice_number', 'po_number', 'doc_date', 'vendor_id', 'vendor_name',
            'currency', 'base_currency', 'fx_rate_doc_to_usd',
            'total_amount', 'total_amount_usd', 'exceptions'
            # lines_json removed
        ]

        self.INV_LINE_COLS = [
            'invoice_number', 'line_number', 'sku_id', 'description',
            'quantity', 'unit_price', 'line_total',
            'unit_price_usd', 'line_total_usd'
        ]

        self.GRN_LINE_COLS = [
            'grn_number', 'po_number', 'doc_date', 'line_number', 'sku_id',
            'description', 'quantity', 'unit_price', 'line_total'
        ]

        # --- 2. Initialize Row Collectors ---
        self.po_agg_rows: List[Dict[str, Any]] = []
        self.po_line_rows: List[Dict[str, Any]] = []
        self.inv_agg_rows: List[Dict[str, Any]] = []
        self.inv_line_rows: List[Dict[str, Any]] = [] # New
        self.grn_line_rows: List[Dict[str, Any]] = []
       
        # --- 3. Initialize Primary Key Trackers ---
        self.seen_po_nums: set = set()
        self.seen_inv_nums: set = set()
        self.seen_grn_nums: set = set()

        # --- 4. Final DataFrames ---
        self.po_aggregate_table: pd.DataFrame = pd.DataFrame(columns=self.PO_AGG_COLS)
        self.po_line_item_table: pd.DataFrame = pd.DataFrame(columns=self.PO_LINE_COLS)
        self.inv_aggregate_table: pd.DataFrame = pd.DataFrame(columns=self.INV_AGG_COLS)
        self.inv_line_item_table: pd.DataFrame = pd.DataFrame(columns=self.INV_LINE_COLS) # New
        self.grn_line_item_table: pd.DataFrame = pd.DataFrame(columns=self.GRN_LINE_COLS)

    def _flatten_common_header(self, norm_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        A helper to flatten common header fields for AGGREGATE tables.
        """
        totals = norm_json.get('totals', {})
        return {
            'doc_date': norm_json.get('doc_date'),
            'vendor_id': norm_json.get('vendor_id'),
            'vendor_name': norm_json.get('vendor_name'),
            'currency': norm_json.get('currency'),
            'base_currency': norm_json.get('base_currency'),
            'fx_rate_doc_to_usd': norm_json.get('fx_rate_doc_to_usd'),
            'total_amount': totals.get('amount_document_ccy'),
            'total_amount_usd': totals.get('amount_usd'),
            'exceptions': "|".join(norm_json.get('exceptions', []))
        }

    def add_document(self, norm_json: Dict[str, Any]):
        """
        Routes the normalized JSON to the correct table processor.
        """
        doc_type = norm_json.get('doc_type')
       
        try:
            if doc_type == 'PO':
                self._process_po(norm_json)
            elif doc_type == 'INVOICE':
                self._process_invoice(norm_json) # Renamed
            elif doc_type == 'GNR':
                self._process_grn_line_item(norm_json)
            else:
                print(f"Warning: Skipping document with unknown type: {doc_type}")
        except Exception as e:
            pk = (
                norm_json.get('po_number') or
                norm_json.get('invoice_number') or
                norm_json.get('grn_number')
            )
            print(f"Error processing document {pk}: {e}")

    def _process_po(self, norm_json: Dict[str, Any]):
        """
        Processes a PO, creating one aggregate row AND N line item rows.
        """
        pk = norm_json.get('po_number')
        if not pk:
            print(f"Warning: Skipping PO with missing 'po_number'.")
            return
        if pk in self.seen_po_nums:
            print(f"Warning: Skipping duplicate PO: {pk}")
            return
           
        # 1. Process Aggregate Row
        common_data = self._flatten_common_header(norm_json)
        agg_row = {'po_number': pk, **common_data}
        self.po_agg_rows.append(agg_row)
        self.seen_po_nums.add(pk)
       
        # 2. Process Line Item Rows
        for line in norm_json.get('lines', []):
            line_row = {
                'po_number': pk,
                'line_number': line.get('line_number'),
                'sku_id': line.get('sku_id'),
                'description': line.get('description'),
                'quantity': line.get('quantity'),
                'unit_price': line.get('unit_price'),
                'line_total': line.get('line_total'),
                'unit_price_usd': line.get('unit_price_usd'),
                'line_total_usd': line.get('line_total_usd'),
            }
            self.po_line_rows.append(line_row)

    def _process_invoice(self, norm_json: Dict[str, Any]): # Renamed
        """
        Processes an Invoice, creating one aggregate row AND N line item rows.
        """
        pk = norm_json.get('invoice_number')
        if not pk:
            print(f"Warning: Skipping Invoice with missing 'invoice_number'.")
            return
        if pk in self.seen_inv_nums:
            print(f"Warning: Skipping duplicate Invoice: {pk}")
            return
           
        # 1. Process Aggregate Row
        common_data = self._flatten_common_header(norm_json)
        agg_row = {
            'invoice_number': pk,
            'po_number': norm_json.get('po_number'),
            **common_data
        }
        self.inv_agg_rows.append(agg_row)
        self.seen_inv_nums.add(pk)

        # 2. Process Line Item Rows (New)
        for line in norm_json.get('lines', []):
            line_row = {
                'invoice_number': pk,
                'line_number': line.get('line_number'),
                'sku_id': line.get('sku_id'),
                'description': line.get('description'),
                'quantity': line.get('quantity'),
                'unit_price': line.get('unit_price'),
                'line_total': line.get('line_total'),
                'unit_price_usd': line.get('unit_price_usd'),
                'line_total_usd': line.get('line_total_usd'),
            }
            self.inv_line_rows.append(line_row)

    def _process_grn_line_item(self, norm_json: Dict[str, Any]):
        """
        Processes a GNR, creating N line item rows.
        Header info is copied into each line row.
        """
        pk = norm_json.get('grn_number')
        if not pk:
            print(f"Warning: Skipping GNR with missing 'grn_number'.")
            return
        if pk in self.seen_grn_nums:
            print(f"Warning: Skipping duplicate GNR: {pk}")
            return
           
        po_num = norm_json.get('po_number')
        doc_date = norm_json.get('doc_date')
       
        lines = norm_json.get('lines', [])
        if not lines:
            print(f"Warning: Skipping GNR {pk} as it has no line items.")
            return

        self.seen_grn_nums.add(pk)
       
        for line in lines:
            line_row = {
                'grn_number': pk,
                'po_number': po_num,
                'doc_date': doc_date,
                'line_number': line.get('line_number'),
                'sku_id': line.get('sku_id'),
                'description': line.get('description'),
                'quantity': line.get('quantity'),
                'unit_price': line.get('unit_price'),
                'line_total': line.get('line_total'),
            }
            self.grn_line_rows.append(line_row)

    # def finalize_tables(self):
    #     """
    #     Converts the collected row lists into five pandas DataFrames.
    #     """
    #     print("Finalizing tables...")
       
    #     if self.po_agg_rows:
    #         self.po_aggregate_table = pd.DataFrame(
    #             self.po_agg_rows, columns=self.PO_AGG_COLS
    #         ).set_index('po_number', drop=True)
           
    #     if self.po_line_rows:
    #         self.po_line_item_table = pd.DataFrame(
    #             self.po_line_rows, columns=self.PO_LINE_COLS
    #         ).set_index(['po_number', 'line_number'], drop=True)

    #     if self.inv_agg_rows:
    #         self.inv_aggregate_table = pd.DataFrame(
    #             self.inv_agg_rows, columns=self.INV_AGG_COLS
    #         ).set_index('invoice_number', drop=True)
       
    #     if self.inv_line_rows: # New
    #         self.inv_line_item_table = pd.DataFrame(
    #             self.inv_line_rows, columns=self.INV_LINE_COLS
    #         ).set_index(['invoice_number', 'line_number'], drop=True)
           
    #     if self.grn_line_rows:
    #         self.grn_line_item_table = pd.DataFrame(
    #             self.grn_line_rows, columns=self.GRN_LINE_COLS
    #         ).set_index(['grn_number', 'line_number'], drop=True)
           
    #     print(f"Processing complete:")
    #     print(f"  - PO_Aggregate:    {len(self.po_aggregate_table)} rows")
    #     print(f"  - PO_LineItem:     {len(self.po_line_item_table)} rows")
    #     print(f"  - INV_Aggregate:   {len(self.inv_aggregate_table)} rows")
    #     print(f"  - INV_LineItem:    {len(self.inv_line_item_table)} rows") # New
    #     print(f"  - GRN_LineItem:    {len(self.grn_line_item_table)} rows")

    def finalize_tables(self):
        """
        Converts the collected row lists into five pandas DataFrames.
        """
        print("Finalizing tables...")

        if self.po_agg_rows:
            self.po_aggregate_table = pd.DataFrame(self.po_agg_rows, columns=self.PO_AGG_COLS)

        if self.po_line_rows:
            self.po_line_item_table = pd.DataFrame(self.po_line_rows, columns=self.PO_LINE_COLS)

        if self.inv_agg_rows:
            self.inv_aggregate_table = pd.DataFrame(self.inv_agg_rows, columns=self.INV_AGG_COLS)

        if self.inv_line_rows:
            self.inv_line_item_table = pd.DataFrame(self.inv_line_rows, columns=self.INV_LINE_COLS)

        if self.grn_line_rows:
            self.grn_line_item_table = pd.DataFrame(self.grn_line_rows, columns=self.GRN_LINE_COLS)

        # Ensure strict column order (matches your screenshot)
        self.po_aggregate_table = self.po_aggregate_table.reindex(columns=self.PO_AGG_COLS)
        self.po_line_item_table = self.po_line_item_table.reindex(columns=self.PO_LINE_COLS)
        self.inv_aggregate_table = self.inv_aggregate_table.reindex(columns=self.INV_AGG_COLS)
        self.inv_line_item_table = self.inv_line_item_table.reindex(columns=self.INV_LINE_COLS)
        self.grn_line_item_table = self.grn_line_item_table.reindex(columns=self.GRN_LINE_COLS)

        # >>> ADD THIS BLOCK RIGHT AFTER THE reindex(...) LINES >>>
        # Enforce exactly one header row per PO / per Invoice
        if isinstance(self.po_aggregate_table, pd.DataFrame) and not self.po_aggregate_table.empty:
            # optional: drop exact duplicates first
            self.po_aggregate_table = self.po_aggregate_table.drop_duplicates().reset_index(drop=True)
            self.po_aggregate_table = _dedupe_header(
                self.po_aggregate_table,
                id_col="po_number",
                date_col="doc_date",
                fx_col="fx_rate_doc_to_usd"
            )

        if isinstance(self.inv_aggregate_table, pd.DataFrame) and not self.inv_aggregate_table.empty:
            self.inv_aggregate_table = self.inv_aggregate_table.drop_duplicates().reset_index(drop=True)
            self.inv_aggregate_table = _dedupe_header(
                self.inv_aggregate_table,
                id_col="invoice_number",
                date_col="doc_date",
                fx_col="fx_rate_doc_to_usd"
            )
# <<< ADD END


        print(f"Processing complete:")
        print(f"  - PO_Aggregate:    {len(self.po_aggregate_table)} rows")
        print(f"  - PO_LineItem:     {len(self.po_line_item_table)} rows")
        print(f"  - INV_Aggregate:   {len(self.inv_aggregate_table)} rows")
        print(f"  - INV_LineItem:    {len(self.inv_line_item_table)} rows")
        print(f"  - GRN_LineItem:    {len(self.grn_line_item_table)} rows")


    def get_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns the five finalized DataFrames.
        """
        return (
            self.po_aggregate_table,
            self.po_line_item_table,
            self.inv_aggregate_table,
            self.inv_line_item_table, # New
            self.grn_line_item_table
        )

