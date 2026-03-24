# Pinterest Visual Search

A context-aware visual image search prototype built as a 14-day academic project.  
Inspired by the *Machine Learning System Design Interview* book.

---

## System Architecture

```
User (Upload Image)
        │
        ▼
  Preprocess (resize, normalize)
        │
        ▼
  CLIP Encoder (image embedding)
        │
        ▼
  FAISS Index (ANN retrieval)  ──> Top-50 Candidates
                                          │
                ┌─────────────────────────┼──────────────────────────┐
                ▼                         ▼                          ▼
        Caption + OCR           GeoNames (place)           Reddit Topics
        (BLIP / EasyOCR)        Wikidata / Wikipedia       (trend signals)
                │                         │                          │
                └─────────────────────────┴──────────────────────────┘
                                          │
                                          ▼
                              Multimodal Reranker
                              (weighted scoring)
                                          │
                                          ▼
                              Top-10 Results (with explanations)
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Image embedding | `CLIP` (openai/clip-vit-base-patch32) |
| Vector search | `FAISS` (IndexFlatIP) |
| Image captioning | `BLIP` (Salesforce/blip-image-captioning-base) |
| OCR | `EasyOCR` |
| Location resolution | `GeoNames` REST API |
| Knowledge enrichment | `Wikidata` SPARQL + `Wikipedia` |
| Trend context | `Reddit` via `PRAW` |
| Backend API | `FastAPI` |
| Frontend | `Streamlit` |

---

## Quick Start

### 1. Clone and setup environment

```bash
git clone <repo-url>
cd PinterestDemo
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
copy .env.example .env
# Edit .env and fill in your Kaggle, GeoNames, and Reddit credentials
```

### 3. Download dataset

```bash
python src/ingestion/download.py
```

### 4. Clean data

```bash
python src/preprocessing/clean.py
```

### 5. Build embeddings

```bash
python scripts/build_embeddings.py
```

### 6. Build FAISS index

```bash
python scripts/build_index.py
```

### 7. Start the API

```bash
uvicorn src.api.main:app --reload
```

### 8. Start the frontend

```bash
streamlit run src/frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Week 2 Context Enrichment (optional)

Run these after the baseline is working:

```bash
# Generate captions and OCR
python scripts/enrich_captions.py

# Landmark detection + GeoNames
python scripts/enrich_locations.py

# Wikidata + Wikipedia entity enrichment
python scripts/enrich_entities.py

# Reddit topic index
python scripts/build_topic_index.py
```

---

## Evaluation

```bash
python scripts/run_evaluation.py
```

Outputs:
- Console table with Recall@K, mAP, nDCG for each mode
- `results/ablation.csv`

---

## Dataset

Kaggle Fashion Product Images (small variant):  
[paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

~44,000 fashion product images with metadata: category, subcategory, color, season, usage.

---

## Project Structure

```
PinterestDemo/
  data/
    raw/              # Downloaded dataset images
    processed/        # Resized/cleaned images
    embeddings/       # CLIP embeddings + FAISS index
    metadata/         # images.csv (enriched progressively)
    eval/             # Evaluation query set
  src/
    ingestion/        # Dataset download + inspection
    preprocessing/    # Image cleaning
    models/           # CLIP, BLIP, OCR wrappers
    indexing/         # FAISS index management
    retrieval/        # Search module
    reranking/        # Metadata + multimodal rerankers
    enrichment/       # GeoNames, Wikidata, Reddit
    api/              # FastAPI backend
    frontend/         # Streamlit app
    evaluation/       # Metrics computation
  scripts/            # Runnable pipeline scripts
  configs/
    config.yaml       # All tunable parameters
  results/
    ablation.csv      # Evaluation comparison table
```

---

## Evaluation Results

| Mode | Recall@10 | mAP | nDCG@10 |
|---|---|---|---|
| Visual only | - | - | - |
| Visual + Metadata | - | - | - |
| Visual + Context | - | - | - |
| Full Context-Aware | - | - | - |

*(Run `scripts/run_evaluation.py` to fill in the table.)*

---

## References

- Nguyen, A. (2023). *Machine Learning System Design Interview*.
- Radford et al. (2021). *CLIP: Learning Transferable Visual Models From Natural Language Supervision*.
- Johnson et al. (2017). *Billion-scale similarity search with GPUs* (FAISS).
- Li et al. (2022). *BLIP: Bootstrapping Language-Image Pre-training*.
