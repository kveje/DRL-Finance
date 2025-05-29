#!/c/Programmer/Python312/python.exe

"""
Cloud setup script for batch experiments.
This script sets up experiments and uploads results to cloud storage.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def upload_results_to_gcs(experiment_name, bucket_name):
    """Upload experiment results to Google Cloud Storage."""
    try:
        from google.cloud import storage
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        experiment_dir = Path(f"experiments/{experiment_name}")
        if not experiment_dir.exists():
            logger.warning(f"Experiment directory {experiment_dir} not found")
            return
        
        # Upload all files in the experiment directory
        for file_path in experiment_dir.rglob("*"):
            if file_path.is_file():
                blob_name = str(file_path.relative_to("experiments"))
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
                logger.info(f"Uploaded {file_path} to gs://{bucket_name}/{blob_name}")
        
        logger.info(f"Successfully uploaded experiment {experiment_name} to cloud storage")
        
    except ImportError:
        logger.warning("google-cloud-storage not installed, uploading with gsutil")
        # Fallback to gsutil
        cmd = f"gsutil -m cp -r experiments/{experiment_name} gs://{bucket_name}/"
        subprocess.run(cmd, shell=True, check=True)
    except Exception as e:
        logger.error(f"Failed to upload results: {e}")

def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def run_experiment(experiment_name, agent_type, interpreter_type, use_bayesian, reward_type, indicator_type, reward_projection_period, price_type, n_episodes):
    """Run a single experiment."""
    logger.info(f"Starting experiment: {experiment_name}")
    
    # Setup experiment
    setup_cmd = [
        sys.executable, "-m", "scripts.setup_experiment",
        "--experiment-name", experiment_name,
        "--agent-type", agent_type,
        "--interpreter-type", interpreter_type,
        "--price-type", price_type,
        "--indicator-type", indicator_type,
        "--reward-type", reward_type,
        "--use-bayesian", str(use_bayesian),
        "--reward-projection-period", str(reward_projection_period),
        "--n-episodes", str(n_episodes)
    ]
    
    logger.info(f"Setting up experiment with command: {' '.join(setup_cmd)}")
    subprocess.run(setup_cmd, check=True)
    
    # Start training
    train_cmd = [
        sys.executable, "-m", "scripts.start_experiment",
        "--experiment-name", experiment_name,
    ]
    
    logger.info(f"Starting training with command: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)
    
    logger.info(f"Completed experiment: {experiment_name}")

def main():
    parser = argparse.ArgumentParser(description="Run cloud experiment")
    parser.add_argument("--experiment-name", required=True, help="Name of experiment")
    parser.add_argument("--agent-type", required=True, help="Agent type (dqn, a2c, ppo, sac)")
    parser.add_argument("--interpreter-type", required=True, help="Interpreter type (discrete, confidence_scaled)")
    parser.add_argument("--use-bayesian", action="store_true", help="Use Bayesian optimization")
    parser.add_argument("--reward-type", required=True, help="Reward type (return, risk_adjusted)")
    parser.add_argument("--indicator-type", default="no", help="Indicator type (no, simple, advanced)")
    parser.add_argument("--reward-projection-period", type=int, default=20, help="Reward projection period")
    parser.add_argument("--price-type", default="both", help="Price type (price, ohlcv, both)")
    parser.add_argument("--n-episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--gcs-bucket", help="Google Cloud Storage bucket for results")
    parser.add_argument("--auto-shutdown", action="store_true", help="Shutdown VM after completion")
    
    args = parser.parse_args()
    
    try:
        # Install dependencies
        install_dependencies()
        
        # Run the experiment
        run_experiment(
            args.experiment_name,
            args.agent_type,
            args.interpreter_type,
            args.use_bayesian,
            args.reward_type,
            args.indicator_type,
            args.reward_projection_period,
            args.price_type,
            args.n_episodes
        )
        
        # Upload results if bucket specified
        if args.gcs_bucket:
            upload_results_to_gcs(args.experiment_name, args.gcs_bucket)
        
        logger.info("All tasks completed successfully!")
        
        # Auto-shutdown if requested
        if args.auto_shutdown:
            logger.info("Auto-shutdown requested, shutting down VM...")
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()