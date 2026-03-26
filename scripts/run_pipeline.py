"""
One-shot pipeline runner for the full baseline setup.

Runs steps in order:
  1. Download dataset
  2. Clean data -> images.csv
  3. Build CLIP embeddings
  4. Build FAISS index

Usage:
    python scripts/run_pipeline.py [--skip-download]
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full data pipeline")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-clean", action="store_true", help="Skip data cleaning")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    args = parser.parse_args()

    config = load_config()

    if not args.skip_download:
        print("\n" + "="*60)
        print("STEP 1: Downloading datasets")
        print("="*60)
        from src.ingestion.download import download_dataset
        download_dataset(config)
    else:
        print("Skipping download.")

    if not args.skip_clean:
        print("\n" + "="*60)
        print("STEP 2: Cleaning data -> images.csv")
        print("="*60)
        from src.preprocessing.clean import clean_all_datasets
        clean_all_datasets(config)
    else:
        print("Skipping data cleaning.")

    if not args.skip_embed:
        print("\n" + "="*60)
        print("STEP 3: Building CLIP embeddings")
        print("="*60)
        from scripts.build_embeddings import build_embeddings
        build_embeddings(config)
    else:
        print("Skipping embedding generation.")

    if not args.skip_index:
        print("\n" + "="*60)
        print("STEP 4: Building FAISS index")
        print("="*60)
        from scripts.build_index import build_index
        build_index(config)
    else:
        print("Skipping index building.")

    print("\n" + "="*60)
    print("Pipeline complete. Start the API with:")
    print("  uvicorn src.api.main:app --reload")
    print("Start the frontend with:")
    print("  streamlit run src/frontend/app.py")
    print("="*60)


if __name__ == "__main__":
    main()
