"""
Streamlit frontend for Pinterest Visual Search.

Run:
    streamlit run src/frontend/app.py
"""

import io
import sys
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Pinterest Visual Search",
    page_icon="🔍",
    layout="wide",
)

# --- Styles ---
st.markdown(
    """
    <style>
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background: #fafafa;
    }
    .score-badge {
        background: #e8f4f8;
        border-radius: 5px;
        padding: 2px 8px;
        font-size: 0.85em;
        color: #1a7abf;
        font-weight: bold;
    }
    .tag-badge {
        background: #f0f0f0;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8em;
        margin: 2px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.title("Pinterest Visual Search")
    st.markdown("Upload an image to find visually similar images.")
    st.divider()

    mode = st.selectbox(
        "Search Mode",
        options=["visual", "reranked", "full"],
        format_func=lambda x: {
            "visual": "Visual Only",
            "reranked": "Visual + Metadata",
            "full": "Full Context-Aware",
        }[x],
    )

    top_k = st.slider("Number of results", min_value=4, max_value=50, value=12, step=4)

    st.divider()
    st.markdown("**API Status**")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success(f"API online")
            st.caption(f"Model: {data.get('model', 'N/A')}")
            st.caption(f"Index: {data.get('index_size', 0):,} images")
            st.caption(f"Device: {data.get('device', 'N/A')}")
        else:
            st.error("API returned non-200 status")
    except Exception:
        st.error("API offline. Start with:\nuvicorn src.api.main:app --reload")

# --- Main area ---
st.header("Visual Image Search")

uploaded_file = st.file_uploader(
    "Upload a query image",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    col_query, col_results = st.columns([1, 3])

    with col_query:
        st.subheader("Query Image")
        query_image = Image.open(uploaded_file)
        st.image(query_image, use_column_width=True)
        st.caption(f"{uploaded_file.name}")

    with col_results:
        st.subheader(f"Similar Images (mode: {mode})")
        with st.spinner("Searching ..."):
            uploaded_file.seek(0)
            try:
                response = requests.post(
                    f"{API_URL}/search",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                    params={"k": top_k, "mode": mode},
                    timeout=30,
                )
                if response.status_code != 200:
                    st.error(f"Search failed: {response.text}")
                    st.stop()
                data = response.json()
                results = data["results"]
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Make sure it is running.")
                st.stop()

        if not results:
            st.info("No results found.")
        else:
            cols_per_row = 4
            for row_start in range(0, len(results), cols_per_row):
                row_results = results[row_start: row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, result in zip(cols, row_results):
                    with col:
                        img_path = result.get("path")
                        if img_path and Path(img_path).exists():
                            try:
                                st.image(Image.open(img_path), use_column_width=True)
                            except Exception:
                                st.image("https://via.placeholder.com/200", use_column_width=True)
                        else:
                            try:
                                img_resp = requests.get(
                                    f"{API_URL}/images/{result['image_id']}",
                                    timeout=5,
                                )
                                if img_resp.status_code == 200:
                                    st.image(
                                        Image.open(io.BytesIO(img_resp.content)),
                                        use_column_width=True,
                                    )
                            except Exception:
                                st.write(f"ID: {result['image_id']}")

                        score = result.get("score", 0)
                        st.markdown(
                            f'<span class="score-badge">Score: {score:.3f}</span>',
                            unsafe_allow_html=True,
                        )

                        if result.get("product_name"):
                            st.caption(result["product_name"][:50])

                        if result.get("category"):
                            st.markdown(
                                f'<span class="tag-badge">{result["category"]}</span>',
                                unsafe_allow_html=True,
                            )

                        if result.get("color"):
                            st.markdown(
                                f'<span class="tag-badge">{result["color"]}</span>',
                                unsafe_allow_html=True,
                            )

                        if result.get("landmark"):
                            st.markdown(
                                f'<span class="tag-badge">📍 {result["landmark"]}</span>',
                                unsafe_allow_html=True,
                            )

                        if result.get("caption"):
                            with st.expander("Caption"):
                                st.write(result["caption"])

                        if result.get("entity_tags"):
                            with st.expander("Context Tags"):
                                st.write(", ".join(result["entity_tags"]))

                        if result.get("explanation"):
                            with st.expander("Why this result?"):
                                for k_exp, v_exp in result["explanation"].items():
                                    st.write(f"**{k_exp}**: {v_exp:.3f}" if isinstance(v_exp, float) else f"**{k_exp}**: {v_exp}")

# --- Comparison mode ---
st.divider()
if st.checkbox("Show side-by-side comparison: Visual vs Full Context"):
    if uploaded_file is not None:
        uploaded_file.seek(0)
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Visual Only")
            try:
                resp_visual = requests.post(
                    f"{API_URL}/search",
                    files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                    params={"k": 8, "mode": "visual"},
                    timeout=30,
                )
                if resp_visual.status_code == 200:
                    visual_results = resp_visual.json()["results"]
                    for r in visual_results[:4]:
                        img_path = r.get("path")
                        if img_path and Path(img_path).exists():
                            st.image(Image.open(img_path), width=150, caption=f"Score: {r['score']:.3f}")
            except Exception as e:
                st.error(str(e))

        with col_b:
            st.subheader("Full Context-Aware")
            uploaded_file.seek(0)
            try:
                resp_full = requests.post(
                    f"{API_URL}/search",
                    files={"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)},
                    params={"k": 8, "mode": "full"},
                    timeout=30,
                )
                if resp_full.status_code == 200:
                    full_results = resp_full.json()["results"]
                    for r in full_results[:4]:
                        img_path = r.get("path")
                        if img_path and Path(img_path).exists():
                            st.image(Image.open(img_path), width=150, caption=f"Score: {r['score']:.3f}")
            except Exception as e:
                st.error(str(e))
    else:
        st.info("Upload an image above to compare modes.")
