# app.py
from pathlib import Path
import base64
import streamlit as st

st.set_page_config(page_title="P2P Demo", page_icon="🧾", layout="wide")

# image_path = Path(__file__).parent / "ema_logo.png"

# # --- Add the image to the sidebar ---
# if image_path.exists():
#     st.sidebar.image(
#         str(image_path),
#         use_column_width=True,
#     )
# else:
#     pass
st.title("Intelligent PO & Invoice Reconciliation Demo")

# st.markdown("""
# <style>
# .block-container {max-width: 1800px; padding-top: 0rem; padding-bottom: 0rem;}
# header, footer {visibility: hidden;}
# .hero-wrap {
#   height: calc(100vh - 2rem);           /* full viewport height minus tiny margin */
#   display: flex; align-items: center; justify-content: center;
# }
# .hero-wrap img {
#   max-width: 100%;
#   max-height: 100%;
#   width: auto; height: auto;
#   object-fit: contain;                  /* keep aspect ratio, no cropping */
#   display: block;
# }
# </style>
# """, unsafe_allow_html=True)
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Use st.image() to display your image
img_path = Path(__file__).with_name("start_app.png")
if img_path.exists():
    st.image(str(img_path))
else:
    pass
# img_path = Path(__file__).with_name("start_app.png")   # <— change if your file is named differently
# if img_path.exists():
#     b64 = base64.b64encode(img_path.read_bytes()).decode()
#     ext = img_path.suffix.lower().lstrip(".") or "png"
#     st.markdown(f"""
#     <div class="hero-wrap">
#       <img src="data:image/{ext};base64,{b64}" alt="hero">
#     </div>
#     """, unsafe_allow_html=True)
# else:
#     st.empty()
st.warning("Please click on Ingestion & Extraction in the sidebar to proceed..")
