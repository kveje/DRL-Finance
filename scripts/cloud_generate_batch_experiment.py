"""
Generate all 32 experiment configurations and GCP deployment scripts.
Run this script to prepare for cloud deployment.
"""

import itertools
import json
from pathlib import Path

def generate_experiment_configs():
    """Generate all experiment configurations."""
    
    # Define the experiment space
    agents = ["dqn", "a2c", "ppo", "sac"]
    interpreters = ["discrete", "confidence_scaled"]
    use_bayesian = [True, False]
    reward_types = ["return", "risk_adjusted"]
    indicator_type = "simple"
    price_type = "both"
    reward_projection_period = 20

    # TEST!!!! REMOVE BEFORE DEPLOYMENT
    return [{"name": "test",
              "agent_type": "dqn", 
              "interpreter_type": "discrete", 
              "use_bayesian": True, 
              "reward_type": "return", 
              "indicator_type": "simple", 
              "reward_projection_period": 20, 
              "price_type": "both",
              "n_episodes": 1000, 
              "max_train_time": 86400}]
    experiments = []
    
    for agent, interpreter, use_bayesian, reward_type in itertools.product(
        agents, interpreters, use_bayesian, reward_types
    ):
        exp_name = f"exp_{agent}_{interpreter}_{use_bayesian}_{reward_type}_indicators"
        
        config = {
            "name": exp_name,
            "agent_type": agent,
            "interpreter_type": interpreter,
            "price_type": price_type,
            "reward_type": reward_type,
            "use_bayesian": use_bayesian,
            "indicator_type": indicator_type,
            "reward_projection_period": reward_projection_period,
            "n_episodes": 1000,
            "max_train_time": 86400,  # 24 hours
        }
        
        experiments.append(config)
    
    return experiments

def load_config():
    """Load configuration from local file or environment variables."""
    config_file = Path("cloud_config.json")
    
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    else:
        # Create template config file
        template_config = {
            "project_id": "your-project-id",
            "zone": "us-central1-a", 
            "machine_type": "n1-standard-4",
            "gpu_type": "nvidia-tesla-t4",
            "gcs_bucket": "your-trading-rl-results",
            "repository_url": "https://github.com/kveje/DRL-Finance.git"
        }
        
        with open(config_file, 'w') as f:
            json.dump(template_config, f, indent=2)
        
        print(f"Created template config: {config_file}")
        print("Please update the values in cloud_config.json before running deployment")
        return template_config

def create_gcp_startup_script(config, gcp_config):
    """Create a startup script for a specific experiment configuration."""
    
    script = f"""#!/bin/bash
set -e

# Log everything
exec > >(tee /var/log/startup-script.log) 2>&1
echo "Starting experiment setup at $(date)"

# Update system and install git
sudo apt-get update -y
sudo apt-get install -y git python3-pip

# Clone repository (replace with your actual repo URL)
echo "Cloning repository..."
git clone {gcp_config['repository_url']} /home/drl_finance
cd /home/drl_finance

# Run the experiment
echo "Starting experiment {config['name']}..."
python3 scripts/cloud_setup.py \\
    --experiment-name "{config['name']}" \\
    --agent-type "{config['agent_type']}" \\
    --interpreter-type "{config['interpreter_type']}" \\
    --price-type "{config['price_type']}" \\
    --indicator-type "{config['indicator_type']}" \\
    --reward-type "{config['reward_type']}" \\
    --reward-projection-period {config['reward_projection_period']} \\
    --n-episodes {config['n_episodes']} \\
    --use-bayesian {config['use_bayesian']} \\
    --gcs-bucket "{gcp_config['gcs_bucket']}" \\
    --auto-shutdown

echo "Experiment completed at $(date)"
"""
    return script

def create_gcp_batch_script():
    """Create the main GCP batch deployment script."""
    
    experiments = generate_experiment_configs()
    gcp_config = load_config()
    
    # Create directories
    Path("cloud_deployment").mkdir(exist_ok=True)
    Path("cloud_deployment/startup_scripts").mkdir(exist_ok=True)
    
    # Create individual startup scripts
    for i, config in enumerate(experiments):
        startup_script = create_gcp_startup_script(config, gcp_config)
        script_file = f"cloud_deployment/startup_scripts/startup_{config['name']}.sh"
        
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Make script executable
        Path(script_file).chmod(0o755)
    
    # Create main deployment script
    deployment_script = f"""#!/bin/bash

# GCP Configuration - UPDATE THESE VALUES
PROJECT_ID="{gcp_config['project_id']}"
ZONE="{gcp_config['zone']}"
MACHINE_TYPE="{gcp_config['machine_type']}"
GPU_TYPE="{gcp_config['gpu_type']}"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
GCS_BUCKET="{gcp_config['gcs_bucket']}"

echo "======================================"
echo "Starting deployment of {len(experiments)} experiments"
echo "Project: $PROJECT_ID"
echo "Zone: $ZONE"
echo "GPU Type: $GPU_TYPE"
echo "Results will be saved to: gs://$GCS_BUCKET"
echo "======================================"

# Verify project setup
echo "Verifying GCP setup..."
if ! gcloud projects describe $PROJECT_ID >/dev/null 2>&1; then
    echo "ERROR: Project $PROJECT_ID not found or not accessible"
    echo "Please update PROJECT_ID in this script"
    exit 1
fi

# Check if bucket exists, create if it doesn't
if ! gsutil ls gs://$GCS_BUCKET >/dev/null 2>&1; then
    echo "Creating GCS bucket: gs://$GCS_BUCKET"
    gsutil mb gs://$GCS_BUCKET
fi

# Create VMs for each experiment
"""

    for i, config in enumerate(experiments):
        vm_name = f"trading-exp-{i:02d}"
        startup_script_path = f"cloud_deployment/startup_scripts/startup_{config['name']}.sh"
        
        deployment_script += f"""
echo "Creating VM {i+1}/{len(experiments)}: {vm_name} for {config['name']}..."
gcloud compute instances create {vm_name} \\
    --project=$PROJECT_ID \\
    --zone=$ZONE \\
    --machine-type=$MACHINE_TYPE \\
    --accelerator="type=$GPU_TYPE,count=1" \\
    --image-family=$IMAGE_FAMILY \\
    --image-project=$IMAGE_PROJECT \\
    --boot-disk-size=100GB \\
    --boot-disk-type=pd-ssd \\
    --maintenance-policy=MIGRATE \\
    --metadata-from-file=startup-script={startup_script_path} \\
    --tags=trading-experiment

if [ $? -eq 0 ]; then
    echo "Created {vm_name}"
else
    echo "Failed to create {vm_name}"
fi

sleep 3  # Small delay between VM creations
"""
    
    deployment_script += f"""
echo "======================================"
echo "All VMs created Monitor progress with:"
echo "  gcloud compute instances list --project=$PROJECT_ID"
echo ""
echo "Check individual VM logs with:"
echo "  gcloud compute instances get-serial-port-output VM_NAME --project=$PROJECT_ID"
echo ""
echo "Download results when complete with:"
echo "  gsutil -m cp -r gs://$GCS_BUCKET/ ./results/"
echo "======================================"
"""
    
    with open("cloud_deployment/deploy_all_experiments.sh", 'w') as f:
        f.write(deployment_script)
    
    # Make script executable
    Path("cloud_deployment/deploy_all_experiments.sh").chmod(0o755)
    
    # Create cleanup script
    cleanup_script = f"""#!/bin/bash

PROJECT_ID="your-project-id"

echo "Stopping and deleting all trading experiment VMs..."

# Get all VMs with the trading-experiment tag
VM_LIST=$(gcloud compute instances list --project=$PROJECT_ID --filter="tags:trading-experiment" --format="value(name,zone)")

if [ -z "$VM_LIST" ]; then
    echo "No trading experiment VMs found"
    exit 0
fi

while IFS=$'\\t' read -r name zone; do
    echo "Deleting VM: $name in zone: $zone"
    gcloud compute instances delete $name --zone=$zone --project=$PROJECT_ID --quiet
done <<< "$VM_LIST"

echo "All trading experiment VMs deleted"
"""
    
    with open("cloud_deployment/cleanup_experiments.sh", 'w') as f:
        f.write(cleanup_script)
    
    Path("cloud_deployment/cleanup_experiments.sh").chmod(0o755)
    
    # Save experiment list for reference
    with open("cloud_deployment/experiments_list.json", 'w') as f:
        json.dump(experiments, f, indent=2)
    
    print(f"Generated {len(experiments)} experiment configurations")
    print("Created files:")
    print("  📁 cloud_deployment/")
    print("    📄 deploy_all_experiments.sh - Main deployment script")
    print("    📄 cleanup_experiments.sh - VM cleanup script")
    print("    📄 experiments_list.json - List of all experiments")
    print("    📁 startup_scripts/ - Individual VM startup scripts")
    print("")
    print("Next steps:")
    print("1. Update PROJECT_ID and GCS_BUCKET in deploy_all_experiments.sh")
    print("2. Update repository URL in startup scripts")
    print("3. Run: chmod +x cloud_deployment/deploy_all_experiments.sh")
    print("4. Run: ./cloud_deployment/deploy_all_experiments.sh")
    
    return experiments

if __name__ == "__main__":
    experiments = create_gcp_batch_script()
    
    print("\\nExperiment list:")
    for exp in experiments:
        print(f"  {exp['name']}")