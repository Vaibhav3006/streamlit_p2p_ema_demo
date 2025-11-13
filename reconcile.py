from __future__ import annotations
import json, math, uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

try:
  from rapidfuzz import fuzz
  _HAVE_RAPIDFUZZ = True
except Exception:
  _HAVE_RAPIDFUZZ = False


# ---------- helpers ----------
def _n(x):
  try:
    if x is None or (isinstance(x, float) and math.isnan(x)):
      return None
    return float(x)
  except Exception:
    return None

def _similar(a: str, b: str) -> float:
  a = (a or "").strip().lower()
  b = (b or "").strip().lower()
  if not a and not b: return 1.0
  if not a or not b: return 0.0
  if _HAVE_RAPIDFUZZ:
    return fuzz.token_set_ratio(a, b) / 100.0
  sa, sb = set(a.split()), set(b.split())
  if not sa or not sb: return 0.0
  return len(sa & sb) / len(sa | sb)

def _pct_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
  if a is None or b is None or a == 0: return None
  return abs(a - b) / abs(a)

def _json(o: Any) -> str:
  try:
    return json.dumps(o, ensure_ascii=False, default=str)
  except Exception:
    return json.dumps(str(o))
 
def _safe_opt_lower_str(x) -> Optional[str]:
  if x is None or (isinstance(x, float) and math.isnan(x)):
    return None
  s = str(x).strip()
  return s.lower() if s else None

def _safe_lower_str(x) -> str:
  if x is None or (isinstance(x, float) and math.isnan(x)):
    return ""
  return str(x).strip().lower()



# ---------- outputs ----------
@dataclass
class ReconRecord:
  invoice_number: str
  po_number: Optional[str]
  vendor_id_match: Optional[bool]
  financial_status: str # MATCHED / MISMATCH / UNKNOWN_PO
  line_status: str # MATCHED / PARTIAL / MISMATCH / NO_LINES / NOT_EVALUATED
  grn_status: str # FULL / PARTIAL / NONE / OVER / NOT_EVALUATED
  overall_status: str # MATCHED / PARTIAL / MISMATCH / UNKNOWN_PO
  match_details: str # JSON string

@dataclass
class ExceptionRecord:
  exception_id: str
  level: str # INVOICE / PO_LINE / GRN_LINE / SYSTEM
  invoice_number: Optional[str]
  po_number: Optional[str]
  grn_number: Optional[str]
  exception_type: str # e.g., PRICE_VARIANCE, QTY_VARIANCE, MISSING_GRN, UNKNOWN_PO, ...
  message: str
  severity: str # INFO / WARN / ERROR
  context_json: str # JSON string


# ---------- reconciler (5-table) ----------
class ReconcilerV2:
  """
  Inputs (DataFrames with exact columns):

  1) PO_Aggregate (po_agg_df)
  - po_number (PK), doc_date, vendor_id, vendor_name, currency,
  total_amount, total_amount_usd, exceptions

  2) PO_LineItem (po_lines_df)
  - po_number (FK), line_number, sku_id, description, quantity, unit_price, line_total

  3) GRN_LineItem (grn_lines_df)
  - grn_number (PK), po_number (FK), doc_date, line_number, 
  - sku_id (column exists but is IGNORED by logic),
  - description (USED FOR MATCHING),
  - quantity (USED FOR MATCHING),
  - unit_price (IGNORED), line_total (IGNORED)

  4) INV_Aggregate (inv_agg_df)
  - invoice_number (PK), po_number (FK), doc_date, vendor_id,
  total_amount, total_amount_usd, lines_json (ignored here), exceptions

  5) INV_LineItem (inv_lines_df)
  - invoice_number (FK), line_number, sku_id, description, quantity, unit_price, line_total

  Returns:
  recon_results_df, exceptions_df
  """

  def __init__(
    self,
    tol_total_pct: float = 0.01, # 1% header tolerance
    tol_price_pct: float = 0.02, # 2% unit price tolerance
    tol_qty_abs: float = 1.0, # +/- 1 unit tolerance
    desc_match_threshold: float = 0.80,
    prefer_usd: bool = True,
  ):
    self.tol_total_pct = tol_total_pct
    self.tol_price_pct = tol_price_pct
    self.tol_qty_abs = tol_qty_abs
    self.desc_match_threshold = desc_match_threshold
    self.prefer_usd = prefer_usd
    self._recons: List[ReconRecord] = []
    self._exceptions: List[ExceptionRecord] = []

  # -------- public API --------
  def reconcile_all(
    self,
    po_agg_df: pd.DataFrame,
    po_lines_df: pd.DataFrame,
    grn_lines_df: pd.DataFrame,
    inv_agg_df: pd.DataFrame,
    inv_lines_df: pd.DataFrame,
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    po_agg_df = po_agg_df.copy()
    po_lines_df = po_lines_df.copy()
    grn_lines_df = grn_lines_df.copy()
    inv_agg_df = inv_agg_df.copy()
    inv_lines_df = inv_lines_df.copy()

    self._flag_duplicate_invoices(inv_agg_df)
    self._flag_duplicate_inv_lines(inv_lines_df)

    # Indexes
    po_header_index = {str(r["po_number"]): r for _, r in po_agg_df.iterrows()}
    po_line_index = self._build_po_line_index(po_lines_df)
    grn_qty_index = self._build_grn_qty_index(grn_lines_df) # Uses NEW logic
    inv_line_index = self._build_inv_line_index(inv_lines_df)

    # Loop invoices
    for _, inv in inv_agg_df.iterrows():
      self._reconcile_one_invoice(inv, po_header_index, po_line_index, grn_qty_index, inv_line_index)

    return (
      pd.DataFrame([r.__dict__ for r in self._recons]),
      pd.DataFrame([e.__dict__ for e in self._exceptions]),
    )

  # -------- single invoice --------
  def _reconcile_one_invoice(
    self,
    inv_row: pd.Series,
    po_header_index: Dict[str, pd.Series],
    po_line_index: Dict[str, List[Dict[str, Any]]],
    grn_qty_index: Dict[Tuple[str, str, Optional[str]], float],
    inv_line_index: Dict[str, List[Dict[str, Any]]],
  ):
    inv_num = str(inv_row.get("invoice_number"))
    po_num = inv_row.get("po_number")
    inv_vendor = inv_row.get("vendor_id")

    # PO lookup
    po_row = po_header_index.get(str(po_num)) if po_num else None
    if po_row is None:
      self._add_exc("INVOICE", inv_num, po_num, "UNKNOWN_PO", "ERROR",
      f"Invoice {inv_num} references unknown PO '{po_num}'",
      ctx={"invoice": inv_num, "po_number": po_num})
      self._push_summary(inv_num, po_num, None, "UNKNOWN_PO", "NOT_EVALUATED", "NOT_EVALUATED", "UNKNOWN_PO",
      details={"reason": "PO not found"})
      return

    # Vendor check
    vendor_match = (str(inv_vendor) == str(po_row.get("vendor_id")))
    if not vendor_match:
      self._add_exc("INVOICE", inv_num, po_num, "VENDOR_MISMATCH", "ERROR",
      "Invoice vendor_id != PO vendor_id",
      ctx={"invoice_vendor_id": inv_vendor, "po_vendor_id": po_row.get("vendor_id")})

    # Financial (header) match
    fin_status, fin_detail = self._financial_match(inv_row, po_row)

    # Line + GRN match using invoice line table
    line_status, grn_status, line_detail = self._line_and_grn_match(
      inv_num, po_num, po_line_index, inv_line_index, grn_qty_index
    )

    overall = self._overall(fin_status, line_status, grn_status)

    self._push_summary(
      inv_num, po_num, vendor_match,
      fin_status, line_status, grn_status, overall,
      details={"financial": fin_detail, "lines": line_detail}
    )

  # -------- financial match --------
  def _financial_match(self, inv: pd.Series, po: pd.Series) -> Tuple[str, Dict[str, Any]]:
    if self.prefer_usd and _n(inv.get("total_amount_usd")) is not None and _n(po.get("total_amount_usd")) is not None:
      inv_total = _n(inv.get("total_amount_usd"))
      po_total = _n(po.get("total_amount_usd"))
      basis = "USD"
    else:
      inv_total = _n(inv.get("total_amount"))
      po_total = _n(po.get("total_amount"))
      basis = po.get("currency") or "DOC_CCY"

    detail = {"basis": basis, "invoice_total": inv_total, "po_total": po_total, "tolerance_pct": self.tol_total_pct}

    if inv_total is None or po_total is None:
      self._add_exc("INVOICE", str(inv.get("invoice_number")), str(po.get("po_number")),
      "TOTAL_MISSING", "ERROR", "Missing totals for financial comparison", ctx=detail)
      return "MISMATCH", detail

    diff_pct = _pct_diff(po_total, inv_total)
    detail["diff_pct"] = diff_pct

    if diff_pct is None or diff_pct <= self.tol_total_pct:
      return "MATCHED", detail

    self._add_exc("INVOICE", str(inv.get("invoice_number")), str(po.get("po_number")),
      "PRICE_VARIANCE", "ERROR",
      f"Invoice total deviates from PO total by {round(diff_pct*100,2)}% (> {self.tol_total_pct*100}%)",
      ctx=detail)
    return "MISMATCH", detail

  # -------- line + grn --------
  def _line_and_grn_match(
    self,
    invoice_number: str,
    po_number: str,
    po_line_index: Dict[str, List[Dict[str, Any]]],
    inv_line_index: Dict[str, List[Dict[str, Any]]],
    grn_qty_index: Dict[Tuple[str, str, Optional[str]], float],
  ) -> Tuple[str, str, Dict[str, Any]]:

    po_lines = po_line_index.get(str(po_number), [])
    inv_lines = inv_line_index.get(str(invoice_number), [])

    if not po_lines and not inv_lines:
      return "NOT_EVALUATED", "NOT_EVALUATED", {"reason": "No PO lines and no Invoice lines"}
    if not inv_lines:
      grn_status = self._evaluate_grn_status_only(po_number, po_lines, grn_qty_index)
      return "NO_LINES", grn_status, {"reason": "Invoice has no itemized lines"}
    if not po_lines:
      # Invoice has lines but PO lines missing: all unmatched
      for il in inv_lines:
        self._add_exc("INVOICE", invoice_number, po_number, "UNMATCHED_INVOICE_LINE", "ERROR",
        "Invoice line cannot be matched (PO has no lines)",
        ctx={"inv_line": il})
      return "MISMATCH", "NOT_EVALUATED", {"reason": "PO has no lines"}

    # Match (SKU → desc)
    matches, inv_unmatched, po_unmatched = self._match_lines(po_lines, inv_lines)

    findings = []
    any_bad = False

    for m in matches:
      pi, ii = po_lines[m["po_idx"]], inv_lines[m["inv_idx"]]

      po_qty, inv_qty = _n(pi.get("quantity")), _n(ii.get("quantity"))
      po_price, inv_price = _n(pi.get("unit_price")), _n(ii.get("unit_price"))

      # price variance
      price_ok, price_diff_pct = True, None
      if po_price not in (None, 0) and inv_price is not None:
        price_diff_pct = abs(inv_price - po_price) / abs(po_price)
        price_ok = price_diff_pct <= self.tol_price_pct
      if not price_ok:
        any_bad = True
        self._add_exc("INVOICE", invoice_number, po_number, "PRICE_VARIANCE", "ERROR",
        "Unit price variance beyond tolerance",
        ctx={"po_line": pi, "inv_line": ii, "price_diff_pct": price_diff_pct})

      # qty vs PO
      qty_ok_po, qty_diff_abs_po = True, None
      if po_qty is not None and inv_qty is not None:
        qty_diff_abs_po = abs(inv_qty - po_qty)
        qty_ok_po = qty_diff_abs_po <= self.tol_qty_abs
      if not qty_ok_po:
        any_bad = True
        self._add_exc("INVOICE", invoice_number, po_number, "QTY_VARIANCE", "ERROR",
        f"Invoiced qty differs from PO by {qty_diff_abs_po} (> {self.tol_qty_abs})",
        ctx={"po_line": pi, "inv_line": ii, "qty_diff_abs": qty_diff_abs_po})

      # qty vs GRN
      # This call now passes sku_id, but the underlying function _lookup_grn_qty will ignore it.
      grn_qty = self._lookup_grn_qty(grn_qty_index, po_number, pi.get("sku_id"), pi.get("description"))
      qty_ok_grn, over_billed = True, False
      if grn_qty is not None and inv_qty is not None:
        if inv_qty - grn_qty > self.tol_qty_abs:
          qty_ok_grn = False
          over_billed = True
          any_bad = True
          self._add_exc("INVOICE", invoice_number, po_number, "OVER_BILLING", "ERROR",
          f"Invoiced qty {inv_qty} exceeds received qty {grn_qty} by > {self.tol_qty_abs}",
          ctx={"po_line": pi, "inv_line": ii, "grn_qty": grn_qty})

      findings.append({
        "match_type": m["match_type"],
        "po_line_number": pi.get("line_number"),
        "po_sku_id": pi.get("sku_id"),
        "po_desc": pi.get("description"),
        "po_qty": po_qty,
        "po_unit_price": po_price,
        "inv_line_number": ii.get("line_number"),
        "inv_sku_id": ii.get("sku_id"),
        "inv_desc": ii.get("description"),
        "inv_qty": inv_qty,
        "inv_unit_price": inv_price,
        "grn_qty": grn_qty,
        "price_diff_pct": price_diff_pct,
        "qty_diff_abs_po": qty_diff_abs_po,
        "price_ok": price_ok,
        "qty_ok_po": qty_ok_po,
        "qty_ok_grn": qty_ok_grn,
        "over_billed": over_billed
      })

    # Unmatched
    for j in inv_unmatched:
      self._add_exc("INVOICE", invoice_number, po_number, "UNMATCHED_INVOICE_LINE", "ERROR",
      "Invoice line could not be matched to any PO line",
      ctx={"inv_line": inv_lines[j]})
      any_bad = True
    for i in po_unmatched:
      self._add_exc("INVOICE", invoice_number, po_number, "UNBILLED_PO_LINE", "WARN",
      "PO line not billed by invoice",
      ctx={"po_line": po_lines[i]})

    # Statuses
    line_status = "MATCHED"
    if not matches:
      line_status = "MISMATCH"
    elif any_bad or len(inv_unmatched) > 0:
      line_status = "PARTIAL"

    grn_status = self._evaluate_grn_status_only(po_number, po_lines, grn_qty_index)

    return line_status, grn_status, {"matches": findings, "inv_unmatched": len(inv_unmatched), "po_unbilled": len(po_unmatched)}

  # -------- match helpers --------
  def _match_lines(self, po_lines, inv_lines):
    matches = []
    po_unmatched = set(range(len(po_lines)))
    inv_unmatched = set(range(len(inv_lines)))

    # SKU-first
    sku_to_po = {}
    for i, pl in enumerate(po_lines):
      k = (pl.get("sku_id") or "").strip().lower()
      if k:
        sku_to_po.setdefault(k, []).append(i)

    for j, il in enumerate(inv_lines):
      k = (il.get("sku_id") or "").strip().lower()
      if not k or k not in sku_to_po: continue
      candidates = [idx for idx in sku_to_po[k] if idx in po_unmatched]
      if candidates:
        i = candidates[0]
        matches.append({"po_idx": i, "inv_idx": j, "match_type": "SKU"})
        po_unmatched.discard(i); inv_unmatched.discard(j)

    # Fuzzy description
    for j in list(inv_unmatched):
      il = inv_lines[j]
      best_i, best_sim = None, -1.0
      for i in list(po_unmatched):
        sim = _similar(po_lines[i].get("description"), il.get("description"))
        if sim > best_sim:
          best_i, best_sim = i, sim
      if best_i is not None and best_sim >= self.desc_match_threshold:
        matches.append({"po_idx": best_i, "inv_idx": j, "match_type": "DESC", "similarity": best_sim})
        po_unmatched.discard(best_i); inv_unmatched.discard(j)

    return matches, inv_unmatched, po_unmatched

  # -------- GRN aggregation --------
 
  def _build_grn_qty_index(self, grn_lines_df: pd.DataFrame):
    """
    **MODIFIED: Builds the GRN index using ONLY description.**
    Ignores sku_id completely.
    """
    idx = {}
    for _, r in grn_lines_df.iterrows():
      po_number = str(r.get("po_number"))
      # Sku is explicitly ignored
      # sku = _safe_opt_lower_str(r.get("sku_id"))
      desc = _safe_lower_str(r.get("description"))
      qty = _n(r.get("quantity")) or 0.0
      
      # Skip if we have no description to match on
      if not desc:
        continue
      
      # Key is (PO_Num, "DESC", description_string)
      k = (po_number, "DESC", desc)
      idx[k] = idx.get(k, 0.0) + qty
    return idx


  def _lookup_grn_qty(self, grn_idx, po_number, sku_id, description):
    """
    **MODIFIED: Looks up GRN quantity using ONLY description.**
    Ignores the passed sku_id.
    """
    # Sku is explicitly ignored
    # sku = _safe_opt_lower_str(sku_id)
    
    desc = _safe_lower_str(description)
    # Can't look up an empty description
    if not desc:
      return None
    
    k = (str(po_number), "DESC", desc)
    return grn_idx.get(k, None)
 
  def _evaluate_grn_status_only(
    self,
    po_number: str,
    po_lines: List[Dict[str, Any]],
    grn_qty_index: Dict[Tuple[str, str, Optional[str]], float],
  ) -> str:
    """
    Summarize GRN coverage for a PO using received quantities vs PO quantities.
    Returns one of: FULL / PARTIAL / NONE / OVER / NOT_EVALUATED
    (This function is unchanged, but its call to _lookup_grn_qty now uses the new logic)
    """
    # No PO lines → nothing to evaluate
    if not po_lines:
      return "NOT_EVALUATED"

    total_po = 0.0
    total_grn = 0.0
    saw_under = False
    saw_over = False

    for pl in po_lines:
      po_qty = _n(pl.get("quantity")) or 0.0
      total_po += po_qty

      grn_qty = self._lookup_grn_qty(
        grn_qty_index,
        po_number,
        pl.get("sku_id"), # This is passed but will be ignored by the new lookup function
        pl.get("description"),
      ) or 0.0
      total_grn += grn_qty

      # Line-level checks with tolerance
      if grn_qty > po_qty + self.tol_qty_abs:
        saw_over = True
      elif grn_qty < po_qty - self.tol_qty_abs:
        saw_under = True

    # Aggregate status
    if total_po <= 0:
      # Nothing ordered → treat as not evaluable
      return "NOT_EVALUATED"

    if total_grn <= self.tol_qty_abs:
      return "NONE"
    if saw_over:
      return "OVER"
    if saw_under:
      return "PARTIAL"
    return "FULL"

  # -------- index builders --------
  def _build_po_line_index(self, po_lines_df: pd.DataFrame):
    idx = {}
    for _, r in po_lines_df.iterrows():
      key = str(r.get("po_number"))
      idx.setdefault(key, []).append({
        "po_number": key,
        "line_number": r.get("line_number"),
        "sku_id": _safe_lower_str(r.get("sku_id")),
        "description": _safe_lower_str(r.get("description")),
        "quantity": _n(r.get("quantity")),
        "unit_price": _n(r.get("unit_price")),
        "line_total": _n(r.get("line_total")),
      })
    return idx


  def _build_inv_line_index(self, inv_lines_df: pd.DataFrame):
    idx = {}
    for _, r in inv_lines_df.iterrows():
      key = str(r.get("invoice_number"))
      idx.setdefault(key, []).append({
        "invoice_number": key,
        "line_number": r.get("line_number"),
        "sku_id": _safe_lower_str(r.get("sku_id")),
        "description": _safe_lower_str(r.get("description")),
        "quantity": _n(r.get("quantity")),
        "unit_price": _n(r.get("unit_price")),
        "line_total": _n(r.get("line_total")),
      })
    return idx


  # -------- duplicates & exceptions --------
  def _flag_duplicate_invoices(self, inv_agg_df: pd.DataFrame):
    dupes = inv_agg_df["invoice_number"].astype(str).duplicated(keep=False)
    for _, r in inv_agg_df[dupes].iterrows():
      self._add_exc("INVOICE", str(r["invoice_number"]), str(r.get("po_number")), "DUPLICATE_INVOICE", "ERROR",
      "Duplicate invoice_number detected", ctx={"invoice_number": r["invoice_number"]})

  def _flag_duplicate_inv_lines(self, inv_lines_df: pd.DataFrame):
    key_cols = ["invoice_number", "line_number"]
    if all(c in inv_lines_df.columns for c in key_cols):
      dupes = inv_lines_df[key_cols].astype(str).duplicated(keep=False)
      for _, r in inv_lines_df[dupes].iterrows():
        self._add_exc("INVOICE", str(r["invoice_number"]), None, "DUPLICATE_INVOICE_LINE", "ERROR",
        "Duplicate (invoice_number, line_number) detected",
        ctx={"invoice_number": r["invoice_number"], "line_number": r["line_number"]})

  def _add_exc(self, level, invoice_number, po_number, etype, severity, message, ctx):
    self._exceptions.append(ExceptionRecord(
      exception_id=str(uuid.uuid4()),
      level=level,
      invoice_number=invoice_number,
      po_number=po_number,
      grn_number=None,
      exception_type=etype,
      message=message,
      severity=severity,
      context_json=_json(ctx)
    ))

  # -------- rollups --------
  def _overall(self, fin_status: str, line_status: str, grn_status: str) -> str:
    if fin_status == "UNKNOWN_PO": return "UNKNOWN_PO"
    bad_fin = fin_status == "MISMATCH"
    bad_line = line_status == "MISMATCH"
    partial_line = line_status == "PARTIAL"
    bad_grn = grn_status in ("NONE", "OVER")
    partial_grn = grn_status == "PARTIAL"
    if not (bad_fin or bad_line or bad_grn or partial_line or partial_grn):
      return "MATCHED"
    if bad_fin or bad_line or bad_grn:
      return "MISMATCH"
    return "PARTIAL"

  def _push_summary(self, invoice_number, po_number, vendor_id_match,
    financial_status, line_status, grn_status, overall, details):
    self._recons.append(ReconRecord(
      invoice_number=invoice_number,
      po_number=po_number,
      vendor_id_match=vendor_id_match,
      financial_status=financial_status,
      line_status=line_status,
      grn_status=grn_status,
      overall_status=overall,
      match_details=_json(details)
    ))

import json
import pandas as pd
from typing import Callable

# 1) Prompt builder for one row
def build_exception_prompt(row: pd.Series) -> str:
    """
    Build a 2–3 line explanation prompt for a single exception row.
    Expected columns: exception_type, message, context_json, invoice_number, po_number (optional).
    """
    exception_type = row.get("exception_type", "")
    message = row.get("message", "")
    context_json = row.get("context_json", "")
    invoice_number = row.get("invoice_number", "")
    po_number = row.get("po_number", "")

    # Make context pretty & safe
    try:
        ctx = json.loads(context_json) if isinstance(context_json, str) else context_json
        ctx_str = json.dumps(ctx, ensure_ascii=False)
    except Exception:
        ctx_str = str(context_json)

    prompt = f"""
You are a financial reconciliation assistant.
Your task is to summarize *why* an exception occurred in 2–3 clear lines for a business analyst.

Information about the exception:
- Invoice number: {invoice_number}
- PO number: {po_number}
- Exception type: {exception_type}
- System message: {message}
- Context (JSON): {ctx_str}

Instructions:
1. Explain the root cause in simple, business-friendly language.
2. Mention which values differ (Invoice vs PO vs GRN) when relevant.
3. Include key numbers (amounts, percentages, quantities) if available.
4. Do not repeat column names; write it as a natural mini-explanation.
5. Output 2–3 short sentences, plain text only (no bullet points, no JSON).
"""
    return prompt.strip()

from openai import OpenAI
import os 
os.environ["OPENAI_API_KEY"] = "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"
# 2) LLM caller – plug in your own model here
# Example using OpenAI's Python client (you can replace with any LLM you use):
#
# from openai import OpenAI
client = OpenAI()
#
def call_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()

# def call_llm(prompt: str) -> str:
#     """
#     Placeholder: replace body with your actual LLM call.
#     For now it just echoes the prompt tail so you don't crash while wiring.
#     """
#     return "[LLM_OUTPUT_PLACEHOLDER] " + prompt[:200] + " ..."


# 3) Apply across dataframe
def add_llm_exception_descriptions(
    df: pd.DataFrame,
    llm_fn: Callable[[str], str] = call_llm,
    prompt_builder: Callable[[pd.Series], str] = build_exception_prompt,
) -> pd.DataFrame:
    """
    For each row in df, call LLM and add column 'Final_Exception_Description'.
    """
    descriptions = []

    for _, row in df.iterrows():
        prompt = prompt_builder(row)
        explanation = llm_fn(prompt)
        descriptions.append(explanation)

    df = df.copy()
    df["Final_Exception_Description"] = descriptions
    return df
