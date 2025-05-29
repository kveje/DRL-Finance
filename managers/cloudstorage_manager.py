import os
import subprocess
import time
import threading
from pathlib import Path
import json

class CloudStorageManager:
    """Manages automatic uploads to cloud storage during training."""
    
    def __init__(self, bucket_name, experiment_name, upload_interval=1800):  # 30 minutes
        self.bucket_name = bucket_name
        self.experiment_name = experiment_name
        self.upload_interval = upload_interval
        self.experiment_dir = Path(f"experiments/{experiment_name}")
        self.is_running = False
        self.upload_thread = None
        
    def start_periodic_uploads(self):
        """Start background thread for periodic uploads."""
        if not self.bucket_name:
            print("No bucket specified, skipping periodic uploads")
            return
            
        self.is_running = True
        self.upload_thread = threading.Thread(target=self._upload_loop)
        self.upload_thread.daemon = True
        self.upload_thread.start()
        print(f"Started periodic uploads every {self.upload_interval//60} minutes")
    
    def stop_periodic_uploads(self):
        """Stop periodic uploads and do final upload."""
        self.is_running = False
        if self.upload_thread:
            self.upload_thread.join()
        
        # Final upload
        self.upload_experiment_results()
        print("Stopped periodic uploads and completed final upload")
    
    def _upload_loop(self):
        """Background upload loop."""
        while self.is_running:
            time.sleep(self.upload_interval)
            if self.is_running:  # Check again after sleep
                try:
                    self.upload_experiment_results()
                    print(f"Periodic upload completed at {time.strftime('%H:%M:%S')}")
                except Exception as e:
                    print(f"Periodic upload failed: {e}")
    
    def upload_experiment_results(self):
        """Upload experiment results to cloud storage."""
        if not self.experiment_dir.exists():
            return
        
        try:
            # Upload entire experiment directory
            cmd = f"gsutil -m rsync -r -d {self.experiment_dir} gs://{self.bucket_name}/{self.experiment_name}/"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Upload failed: {e}")
            # Fallback: try individual file uploads
            self._upload_individual_files()
    
    def _upload_individual_files(self):
        """Fallback: upload files individually."""
        for file_path in self.experiment_dir.rglob("*"):
            if file_path.is_file():
                try:
                    blob_name = f"{self.experiment_name}/{file_path.relative_to(self.experiment_dir)}"
                    cmd = f"gsutil cp {file_path} gs://{self.bucket_name}/{blob_name}"
                    subprocess.run(cmd, shell=True, check=True, capture_output=True)
                except Exception as e:
                    print(f"Failed to upload {file_path}: {e}")