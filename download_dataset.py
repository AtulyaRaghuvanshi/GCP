"""Download the GCP dataset from Google Drive and extract it locally.

This script uses gdown to fetch the shared folder, then returns the
resolved paths to train_dataset and test_dataset so they can be passed
directly to eda.py / train.py / infer.py.

Usage (standalone):
    python download_dataset.py

Usage (imported by other scripts - automatic download if dirs missing):
    from download_dataset import ensure_dataset
    train_dir, test_dir = ensure_dataset()
"""

import os
import sys
import shutil
import zipfile
import subprocess

# ─── CONFIGURE HERE ────────────────────────────────────────────────────────────
# The Google Drive FOLDER id - taken from the shared link:
# https://drive.google.com/drive/folders/1RDNiAO1EynKrRDomcVNXQW1-ct5zzvE5
GDRIVE_FOLDER_ID = "1RDNiAO1EynKrRDomcVNXQW1-ct5zzvE5"

# Where to download and keep the data on this machine.
# Change to e.g. "/tmp/gcp_data" if you want it wiped on reboot,
# or to "/content/gcp_data" if running in Google Colab.
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Local Google Drive ZIP parts. These are preferred over gdown when present.
DEFAULT_ZIP_PATHS = [
    r"D:\Downloads\train_dataset-20260614T100421Z-3-001.zip",
    r"D:\Downloads\train_dataset-20260614T100421Z-3-002.zip",
    r"D:\Downloads\test_dataset-20260614T100423Z-3-001.zip",
]

# Expected sub-directory names inside the downloaded folder.
TRAIN_SUBDIR = "train_dataset"
TEST_SUBDIR  = "test_dataset"
# ───────────────────────────────────────────────────────────────────────────────


def _install_gdown():
    """Install gdown if it is not already available."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[download] gdown not found - installing via pip ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"])


def _download_folder(folder_id: str, dest_dir: str):
    """Download a public Google Drive folder into dest_dir."""
    import gdown
    os.makedirs(dest_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"[download] Downloading Google Drive folder -> {dest_dir}")
    print(f"           URL: {url}")
    # gdown's folder download puts contents directly under dest_dir
    gdown.download_folder(url=url, output=dest_dir, quiet=False, use_cookies=False)


def _safe_extract_zip(zip_path: str, dest_dir: str):
    """Extract zip_path into dest_dir without allowing paths outside dest_dir."""
    dest_dir = os.path.abspath(dest_dir)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = os.path.abspath(os.path.join(dest_dir, member.filename))
            if not target.startswith(dest_dir + os.sep) and target != dest_dir:
                raise RuntimeError(f"Unsafe path in zip {zip_path}: {member.filename}")
        zf.extractall(dest_dir)


def _extract_local_zips(zip_paths: list[str], dest_dir: str) -> bool:
    """Extract all existing local dataset ZIPs. Return True if any were found."""
    existing = [path for path in zip_paths if os.path.isfile(path)]
    if not existing:
        return False

    missing = [path for path in zip_paths if not os.path.isfile(path)]
    if missing:
        print("[download] Some configured local ZIPs were not found:")
        for path in missing:
            print(f"           missing: {path}")

    os.makedirs(dest_dir, exist_ok=True)
    for zip_path in existing:
        print(f"[download] Extracting local ZIP -> {zip_path}")
        _safe_extract_zip(zip_path, dest_dir)
    return True


def _find_subdir(root: str, name: str) -> str | None:
    """Search up to 2 levels under root for a directory called `name`."""
    if not os.path.isdir(root):
        return None

    # Direct child
    candidate = os.path.join(root, name)
    if os.path.isdir(candidate):
        return candidate
    # One level deeper (gdown sometimes creates an extra wrapper folder)
    for child in os.listdir(root):
        candidate = os.path.join(root, child, name)
        if os.path.isdir(candidate):
            return candidate
    return None


def ensure_dataset(data_dir: str = DEFAULT_DATA_DIR, force: bool = False, zip_paths: list[str] | None = None):
    """Return (train_dir, test_dir), downloading from Drive if needed.

    Args:
        data_dir:  Local directory where the dataset is stored / will be saved.
        force:     If True, re-download even if the dataset already exists.

    Returns:
        (train_dir, test_dir) - absolute paths ready to pass to other scripts.
    """
    train_dir = _find_subdir(data_dir, TRAIN_SUBDIR)
    test_dir  = _find_subdir(data_dir, TEST_SUBDIR)

    already_present = (
        train_dir and os.path.isdir(train_dir) and
        test_dir  and os.path.isdir(test_dir)
    )

    if already_present and not force:
        print(f"[download] Dataset already present - skipping download.")
        print(f"           train -> {train_dir}")
        print(f"           test  -> {test_dir}")
        return train_dir, test_dir

    extracted = _extract_local_zips(zip_paths or DEFAULT_ZIP_PATHS, data_dir)
    if not extracted:
        _install_gdown()
        _download_folder(GDRIVE_FOLDER_ID, data_dir)

    # Resolve paths after extraction / download
    train_dir = _find_subdir(data_dir, TRAIN_SUBDIR)
    test_dir  = _find_subdir(data_dir, TEST_SUBDIR)

    if not train_dir:
        raise RuntimeError(
            f"Could not find '{TRAIN_SUBDIR}' under {data_dir} after download.\n"
            f"Contents: {os.listdir(data_dir)}\n"
            f"If the folder structure differs, set TRAIN_SUBDIR at the top of download_dataset.py."
        )
    if not test_dir:
        raise RuntimeError(
            f"Could not find '{TEST_SUBDIR}' under {data_dir} after download.\n"
            f"Contents: {os.listdir(data_dir)}"
        )

    print(f"[download] Ready.")
    print(f"           train -> {train_dir}")
    print(f"           test  -> {test_dir}")
    return train_dir, test_dir


def cleanup(data_dir: str = DEFAULT_DATA_DIR):
    """Delete the local dataset directory (to free space when done)."""
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
        print(f"[download] Removed {data_dir}")
    else:
        print(f"[download] Nothing to clean up at {data_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download GCP dataset from Google Drive")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help=f"Where to store the dataset (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if dataset already exists")
    parser.add_argument("--zip", dest="zip_paths", action="append",
                        help="Local dataset ZIP to extract. Can be passed multiple times.")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete the local dataset directory instead of downloading")
    args = parser.parse_args()

    if args.cleanup:
        cleanup(args.data_dir)
    else:
        train_dir, test_dir = ensure_dataset(
            args.data_dir,
            force=args.force,
            zip_paths=args.zip_paths,
        )
        print("\nNext steps:")
        print(f"  python eda.py   --train_dir \"{train_dir}\" --test_dir \"{test_dir}\"")
        print(f"  python train.py --train_dir \"{train_dir}\" --output_dir ./checkpoints")
        print(f"  python infer.py --test_dir  \"{test_dir}\"  --checkpoint ./checkpoints/best_model.pt --output predictions.json")
