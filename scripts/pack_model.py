import argparse
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pyspark


def zipdir(path: Path, ziph: zipfile.ZipFile):
    for root, dirs, files in os.walk(path):
        for f in files:
            p = Path(root) / f
            ziph.write(p, p.relative_to(path.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Pack saved model dir into dist/*.zip with metadata"
    )
    parser.add_argument(
        "--model-dir",
        default="models/spam_clf",
        help="Directory with saved PipelineModel",
    )
    parser.add_argument(
        "--out", default="dist/spam_clf.zip", help="Path to resulting zip"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"Model dir not found: {model_dir}")

    dist_dir = Path(args.out).parent
    dist_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_dir": str(model_dir),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spark_version": pyspark.__version__,
        "java_version": "17",
        "pipeline_type": "Text TF-IDF + LogisticRegression",
    }
    meta_path = model_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out_zip = Path(args.out)
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipdir(model_dir, zipf)

    print(f"ZIP OK: {out_zip} ({out_zip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
