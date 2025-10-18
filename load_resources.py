import argparse
import os
import zipfile
from pathlib import Path

import gdown

TRAIN_URL = "https://drive.google.com/file/d/131PxXNyMNZj1ldEWXP-NQBDXHI3BPlln/view?usp=sharing"
INFERENCE_URL = "https://drive.google.com/file/d/1cY5He9zUtol1lUbiCnlAtvQ4qmJ9mKZO/view?usp=sharing"

parser = argparse.ArgumentParser(
    description="Download all necessary resources for recreation and testing"
)
parser.add_argument("--output", "-o", type=str, required=True, help="Output dir f" "or resources")
parser.add_argument(
    "--inference_only",
    "-i",
    action="store_true",
    help="Flag to specify if only inference utilities are needed to be fetched",
)


def main(args):
    if args.inference_only:
        URL = INFERENCE_URL
    else:
        URL = TRAIN_URL
    os.makedirs(args.output, exist_ok=True)
    output = Path(args.output)
    archive_name = "inference.zip" if args.inference_only else "resources.zip"
    print(f"Downloading from: {URL}")
    print(f"Saving to: {output.absolute().resolve()}")

    os.makedirs(args.output, exist_ok=True)
    zip_path = output / archive_name
    gdown.download(url=URL, output=str(zip_path), quiet=False, use_cookies=False, fuzzy=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(args.output)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
