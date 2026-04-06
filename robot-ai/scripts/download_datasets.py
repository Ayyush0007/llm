"""
download_datasets.py — Download Indian road datasets from Roboflow Universe.

Downloads and merges:
  1. Pothole Detection dataset  (~6,000 images)
  2. Cow Detection dataset       (~1,500 images)
  3. Auto-rickshaw dataset        (~800 images)

Usage:
    pip install roboflow
    python3 scripts/download_datasets.py --rf-key YOUR_ROBOFLOW_API_KEY

Get your free API key at: https://app.roboflow.com → Settings → Roboflow API
"""

import os
import shutil
import argparse


DATASETS_DIR = "/Users/yashmogare/robot-ai/llm/robot-ai/datasets"

ROBOFLOW_DATASETS = [
    {
        "workspace" : "roboflow-100",
        "project"   : "pothole-detection-dc9ex",
        "version"   : 2,
        "name"      : "pothole",
    },
    {
        "workspace" : "cattle-g0giy",
        "project"   : "cattle-detection-g0giy",
        "version"   : 1,
        "name"      : "cow",
    },
    {
        "workspace" : "final-project-wnmrb",
        "project"   : "auto-rickshaw-detection",
        "version"   : 1,
        "name"      : "auto_rickshaw",
    },
]


def download_all(api_key: str):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ roboflow not installed. Run: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)

    for ds in ROBOFLOW_DATASETS:
        print(f"\n📥 Downloading '{ds['name']}' dataset...")
        try:
            project  = rf.workspace(ds["workspace"]).project(ds["project"])
            version  = project.version(ds["version"])
            dataset  = version.download(
                "yolov8",
                location=os.path.join(DATASETS_DIR, ds["name"])
            )
            print(f"✅ '{ds['name']}' saved to {dataset.location}")
        except Exception as e:
            print(f"⚠️  Could not download '{ds['name']}': {e}")
            print(f"   Try manually at: https://universe.roboflow.com")

    print("\n🔗 Merging datasets into combined training set...")
    _merge_datasets()
    print("✅ All datasets downloaded and merged!")
    print(f"   Location: {DATASETS_DIR}/combined/")


def _merge_datasets():
    """Copy all images + labels into a single combined/ folder."""
    combined = os.path.join(DATASETS_DIR, "combined")
    for split in ["train", "valid", "test"]:
        os.makedirs(os.path.join(combined, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(combined, split, "labels"), exist_ok=True)

    for ds in ROBOFLOW_DATASETS:
        ds_dir = os.path.join(DATASETS_DIR, ds["name"])
        if not os.path.exists(ds_dir):
            continue
        for split in ["train", "valid", "test"]:
            for kind in ["images", "labels"]:
                src = os.path.join(ds_dir, split, kind)
                dst = os.path.join(combined, split, kind)
                if not os.path.exists(src):
                    continue
                for fname in os.listdir(src):
                    # Prefix files with dataset name to avoid name conflicts
                    src_file = os.path.join(src, fname)
                    dst_file = os.path.join(dst, f"{ds['name']}_{fname}")
                    shutil.copy2(src_file, dst_file)


def print_instructions():
    """Print manual download instructions if no API key provided."""
    print("\n📋 Manual Download Instructions:")
    print("=" * 50)
    for ds in ROBOFLOW_DATASETS:
        print(f"\n{ds['name'].upper()}:")
        print(f"  https://universe.roboflow.com/{ds['workspace']}/{ds['project']}")
        print(f"  → Click 'Download' → Choose 'YOLOv8' format")
        print(f"  → Save to: {DATASETS_DIR}/{ds['name']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Roboflow datasets")
    parser.add_argument(
        "--rf-key", default=None,
        help="Roboflow API key (get free at https://app.roboflow.com)"
    )
    args = parser.parse_args()

    if args.rf_key:
        download_all(args.rf_key)
    else:
        print("⚠️  No API key provided — showing manual instructions.")
        print_instructions()
        print("\n💡 To use API: python3 scripts/download_datasets.py --rf-key YOUR_KEY")
