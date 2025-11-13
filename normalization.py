# Patch: modify FX lookup to use a single (latest) exchange rate per currency pair,
# regardless of the document date. Reuse the same CSV previously written.
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from datetime import date
from typing import Optional
import pandas as pd
from pathlib import Path

# Re-import or reuse helpers from earlier cell if present
import re
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any, List

# Set decimal precision
getcontext().prec = 28
TWOPL = Decimal("0.01")

# --- Helper Functions (from your code) ---
def _q(x): return Decimal(str(x)) if x is not None else Decimal("0")
def _q2(x: Decimal) -> Decimal: return x.quantize(TWOPL, rounding=ROUND_HALF_UP)
def _clean_vendor(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).title()
def _slug_vendor(name: str) -> str:
    s = re.sub(r"\s+", " ", name or "").strip().upper()
    return re.sub(r"[^A-Z0-9]+", "-", s).strip("-") or "UNKNOWN"

def _normalize_two_digit_year(dt):
    from datetime import date as _date
    if dt.year < 1990:
        return _date(dt.year + 100, dt.month, dt.day)
    return dt

def _try_strptime_many(d: str, fmts: list[str]):
    from datetime import datetime
    for f in fmts:
        try:
            parsed = datetime.strptime(d, f).date()
            return _normalize_two_digit_year(parsed)
        except Exception:
            continue
    return None

CURRENCY_DATE_POLICY = {
    "USD": "monthfirst", "CAD": "monthfirst", "MXN": "monthfirst",
    "EUR": "dayfirst", "GBP": "dayfirst", "INR": "dayfirst", "AUD": "dayfirst",
    "NZD": "dayfirst", "CHF": "dayfirst", "SEK": "dayfirst", "NOK": "dayfirst",
    "DKK": "dayfirst", "ZAR": "dayfirst", "BRL": "dayfirst", "SGD": "dayfirst",
    "JPY": "yearfirst", "CNY": "yearfirst", "KRW": "yearfirst",
}

def _parse_date_by_currency(d: str, currency: str):
    from pandas import to_datetime
    d = (str(d) or "").strip()
    ccy = (currency or "").upper().strip()
    policy = CURRENCY_DATE_POLICY.get(ccy, "dayfirst")
    MONTH_FIRST = ["%m/%d/%Y","%m/%d/%y","%m-%d-%Y","%m-%d-%y","%m.%d.%Y","%m.%d.%y"]
    DAY_FIRST   = ["%d/%m/%Y","%d/%m/%y","%d-%m-%Y","%d-%m-%y","%d.%m.%Y","%d.%m.%y"]
    YEAR_FIRST  = ["%Y-%m-%d","%Y/%m/%d","%Y.%m.%d","%y-%m-%d","%y/%m/%d","%y.%m.%d"]
    
    if policy == "monthfirst":
        out = _try_strptime_many(d, MONTH_FIRST + DAY_FIRST + YEAR_FIRST)
    elif policy == "yearfirst":
        out = _try_strptime_many(d, YEAR_FIRST + DAY_FIRST + MONTH_FIRST)
    else:
        out = _try_strptime_many(d, DAY_FIRST + MONTH_FIRST + YEAR_FIRST)
    
    if out: return out
    
    # Fallback to pandas
    try:
        if policy == "monthfirst":
            return _normalize_two_digit_year(to_datetime(d, dayfirst=False, errors="raise").date())
        if policy == "yearfirst":
            return _normalize_two_digit_year(to_datetime(d, yearfirst=True, errors="raise").date())
        return _normalize_two_digit_year(to_datetime(d, dayfirst=True, errors="raise").date())
    except Exception:
         # Final fallback
         return _normalize_two_digit_year(to_datetime(d, errors="raise").date())


# --- Patched FXLookup Class (from your code) ---
class FXLookupSingle:
    """
    Use the latest available rate in the CSV for each (from,to) pair,
    ignoring the document date.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["rate_date"] = pd.to_datetime(self.df["rate_date"]).dt.date
        self.df["from_currency"] = self.df["from_currency"].str.upper().str.strip()
        self.df["to_currency"] = self.df["to_currency"].str.upper().str.strip()
        self.df["rate"] = self.df["rate"].apply(lambda x: Decimal(str(x)))
        self.latest = (
            self.df.sort_values("rate_date")
                    .groupby(["from_currency","to_currency"], as_index=False)
                    .tail(1)
        )

    @classmethod
    def from_csv(cls, path: str | Path) -> "FXLookupSingle":
        return cls(pd.read_csv(path, dtype={"from_currency": str, "to_currency": str}))

    def get_rate(self, _on_date_ignored, from_ccy: str, to_ccy: str = "USD") -> Optional[Decimal]:
        from_ccy = (from_ccy or "").upper().strip()
        to_ccy = (to_ccy or "").upper().strip()
        if from_ccy == to_ccy:
            return Decimal("1")
        sub = self.latest[(self.latest["from_currency"]==from_ccy) & (self.latest["to_currency"]==to_ccy)]
        if sub.empty:
            return None
        return Decimal(sub.iloc[0]["rate"])

# ==============================================================================
# === THE NEW GENERIC NORMALIZATION FUNCTION ===
# ==============================================================================
def normalize_document_advanced(
    doc_json: Dict[str, Any], 
    fx_csv_path: str | Path
) -> Dict[str, Any]:
    """
    Normalizes a PO, Invoice, or GNR from a raw JSON dict,
    using advanced validation and the FXLookupSingle class.
    """
    exceptions: List[str] = []

    fx_path = Path(fx_csv_path)
    if not fx_path.is_absolute():
        # assume the file is in a "Data" folder at repo root
        fx_path = Path(__file__).parent.parent / "Data" / fx_path
    if not fx_path.exists():
        raise FileNotFoundError(f"FX rate file not found: {fx_path}")

    # === 1. Document Type and Number Normalization ===
    doc_type_raw = (doc_json.get("document_type") or "UNKNOWN").strip().upper()
    
    # Standardize doc_type
    if doc_type_raw in {"PO", "PURCHASE ORDER"}:
        doc_type = "PO"
    elif doc_type_raw in {"INV", "INVOICE"}:
        doc_type = "INVOICE"
    elif doc_type_raw in {"GNR", "GRN", "GOODS RECEIPT NOTE"}:
        doc_type = "GNR"
    else:
        doc_type = doc_type_raw
        exceptions.append(f"UNEXPECTED_DOC_TYPE:{doc_type}")
    
    # Get all possible numbers
    po_num = doc_json.get("po_number")
    inv_num = doc_json.get("inv_number")
    grn_num = doc_json.get("grn_number")

    # === 2. Common Field Normalization ===
    try:
        doc_date = _parse_date_by_currency(str(doc_json.get("date","")), doc_json.get("currency","USD"))
    except Exception:
        exceptions.append(f"BAD_DATE:{doc_json.get('date')}")
        from datetime import date as _date
        doc_date = _date.today()

    vendor_name_raw = doc_json.get("vendor_name") or ""
    vendor_name = _clean_vendor(vendor_name_raw)
    vendor_id = _slug_vendor(vendor_name_raw)
    ccy = (doc_json.get("currency") or "USD").upper().strip()

    # === 3. Line Item & Total Normalization (This is already generic) ===
    items = doc_json.get("items") or []
    if items is None: # Handle 'items': null case
        items = []
        
    norm_lines: List[Dict[str, Any]] = []
    running_total = Decimal("0")
    
    for i, it in enumerate(items, start=1):
        if not isinstance(it, dict): # Skip bad line items
             exceptions.append(f"INVALID_LINE_ITEM_FORMAT#line={i}: item was not a dict")
             continue
             
        qty = _q(it.get("quantity"))
        unit_price = _q(it.get("unit_price"))
        expected_line_total = _q2(qty * unit_price)
        provided_line_total = _q(it.get("line_total"))
        provided_line_total_q2 = _q2(provided_line_total)
        
        if provided_line_total_q2 != expected_line_total:
            # Allow for small rounding differences
            if (provided_line_total_q2 - expected_line_total).copy_abs() <= Decimal("0.01"):
                provided_line_total_q2 = expected_line_total
            else:
                exceptions.append(f"LINE_TOTAL_MISMATCH#line={i}: provided={provided_line_total_q2} expected={expected_line_total}")
        
        norm_lines.append({
            "line_number": i,
            "sku_id": it.get("sku_id"),
            "description": (it.get("name") or "").strip(),
            "quantity": qty,
            "unit_price": _q2(unit_price),
            "line_total": provided_line_total_q2,
        })
        running_total += provided_line_total_q2
    
    running_total = _q2(running_total)

    provided_total = _q2(_q(doc_json.get("total_amount")))
    
    # Only validate total if it was provided
    if provided_total != Decimal("0") and provided_total != running_total:
        if (provided_total - running_total).copy_abs() <= Decimal("0.02"):
            provided_total = running_total # Snap to computed total
        else:
            exceptions.append(f"HEADER_TOTAL_MISMATCH: provided={provided_total} computed={running_total}")
    
    # Use the computed total if header total was missing, otherwise use (snapped) header total
    gross_total = running_total if provided_total == Decimal("0") else provided_total

    # === 4. FX Conversion (This is already generic) ===
    # We initialize the lookup only once for this function call
    try:
        fx = FXLookupSingle.from_csv(fx_csv_path)
        rate = fx.get_rate(None, ccy, "USD")
    except Exception as e:
        exceptions.append(f"FX_LOOKUP_FAILED: {e}")
        fx = None
        rate = None

    if rate is None:
        if ccy != "USD": # Don't add exception if it's already USD
            exceptions.append(f"MISSING_FX_RATE:{ccy}->USD@LATEST")
        rate = Decimal("1") if ccy == "USD" else Decimal("0")

    for line in norm_lines:
        if rate == 0:
            line["unit_price_usd"] = None
            line["line_total_usd"] = None
        else:
            line["unit_price_usd"] = _q2(line["unit_price"] * rate)
            line["line_total_usd"] = _q2(line["line_total"] * rate)

    total_usd = None if rate == 0 else _q2(gross_total * rate)

    # === 5. Build Final Standardized Output ===
    return {
        "doc_type": doc_type,
        "po_number": po_num,
        "invoice_number": inv_num,
        "grn_number": grn_num,
        "doc_date": doc_date.isoformat(),
        "vendor_id": vendor_id,
        "vendor_name": vendor_name,
        "currency": ccy,
        "base_currency": "USD",
        "fx_policy": "single_latest_rate",
        "fx_rate_doc_to_usd": None if rate == 0 else str(rate),
        "totals": {
            "amount_document_ccy": str(gross_total),
            "amount_usd": None if total_usd is None else str(total_usd),
        },
        "lines": [
            {
                "line_number": ln["line_number"],
                "sku_id": ln["sku_id"],
                "description": ln["description"],
                "quantity": str(ln["quantity"]),
                "unit_price": str(ln["unit_price"]),
                "line_total": str(ln["line_total"]),
                "unit_price_usd": None if ln.get("unit_price_usd") is None else str(ln["unit_price_usd"]),
                "line_total_usd": None if ln.get("line_total_usd") is None else str(ln["line_total_usd"]),
            }
            for ln in norm_lines
        ],
        "exceptions": exceptions,
    }
