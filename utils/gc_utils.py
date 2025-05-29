"""Cloud utilities for uploading results."""

import os
import subprocess
from pathlib import Path

def upload_to_gcs(local_path, bucket_name, blob_name):
    """Upload file to Google Cloud Storage using gsutil."""
    cmd = f"gsutil cp {local_path} gs://{bucket_name}/{blob_name}"
    subprocess.run(cmd, shell=True, check=True)

def sync_directory_to_gcs(local_dir, bucket_name, prefix=""):
    """Sync entire directory to GCS."""
    cmd = f"gsutil -m rsync -r {local_dir} gs://{bucket_name}/{prefix}"
    subprocess.run(cmd, shell=True, check=True)