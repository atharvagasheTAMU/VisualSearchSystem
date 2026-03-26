"""Download all configured datasets from Kaggle into their raw subdirectories."""

import os
import subprocess
import zipfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _ensure_kaggle_credentials() -> None:
    kaggle_username = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")
    if not kaggle_username or not kaggle_key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in .env.\n"
            "Get your API key at: https://www.kaggle.com/settings/account"
        )
    kaggle_json_dir = Path.home() / ".kaggle"
    kaggle_json_dir.mkdir(exist_ok=True)
    kaggle_json_path = kaggle_json_dir / "kaggle.json"
    if not kaggle_json_path.exists():
        kaggle_json_path.write_text(
            f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}'
        )
        kaggle_json_path.chmod(0o600)


def _download_one(kaggle_slug: str, raw_dir: Path) -> None:
    """Download and extract one Kaggle dataset into raw_dir."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "dataset.zip"
    if zip_path.exists():
        print(f"  Archive already exists at {zip_path}, skipping download.")
    else:
        print(f"  Downloading {kaggle_slug} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_slug, "-p", str(raw_dir)],
            check=True,
        )
        downloaded = list(raw_dir.glob("*.zip"))
        if downloaded:
            downloaded[0].rename(zip_path)

    # Check if already extracted (any subdirectory or image file present)
    contents = [p for p in raw_dir.iterdir() if p != zip_path]
    if contents:
        print(f"  Already extracted ({len(contents)} items in {raw_dir}).")
        return

    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"  Extraction complete -> {raw_dir}")


def download_dataset(config: dict) -> None:
    """Download all datasets listed in config['datasets']."""
    _ensure_kaggle_credentials()

    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets defined in config['datasets'].")

    for ds in datasets:
        name = ds["name"]
        slug = ds["kaggle_slug"]
        raw_dir = Path(ds["raw_dir"])
        if not raw_dir.is_absolute():
            raw_dir = Path("data") / raw_dir  # resolve relative to project root
        print(f"\n[{name}] {slug} -> {raw_dir}")
        _download_one(slug, raw_dir)


if __name__ == "__main__":
    cfg = load_config()
    download_dataset(cfg)
