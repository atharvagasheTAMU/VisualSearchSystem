"""Download the Kaggle fashion dataset into data/raw/."""

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


def download_dataset(config: dict) -> None:
    kaggle_dataset = config["dataset"]["kaggle_dataset"]
    raw_dir = Path(config["paths"]["raw_images"])
    raw_dir.mkdir(parents=True, exist_ok=True)

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

    zip_path = raw_dir / "dataset.zip"
    if zip_path.exists():
        print(f"Archive already exists at {zip_path}, skipping download.")
    else:
        print(f"Downloading dataset: {kaggle_dataset} ...")
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                kaggle_dataset,
                "-p",
                str(raw_dir),
            ],
            check=True,
        )
        downloaded = list(raw_dir.glob("*.zip"))
        if downloaded:
            downloaded[0].rename(zip_path)

    images_dir = raw_dir / "images"
    if images_dir.exists() and any(images_dir.iterdir()):
        print(f"Images already extracted at {images_dir}.")
        return

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    print(f"Extraction complete. Files in {raw_dir}:")
    for item in sorted(raw_dir.iterdir())[:10]:
        print(f"  {item.name}")


if __name__ == "__main__":
    cfg = load_config()
    download_dataset(cfg)
