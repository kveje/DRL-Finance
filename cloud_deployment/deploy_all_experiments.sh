#!/bin/bash

# GCP Configuration - UPDATE THESE VALUES
PROJECT_ID="finrl-461213"
ZONE="europe-west1-b"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
IMAGE_FAMILY="pytorch-latest-gpu"
IMAGE_PROJECT="deeplearning-platform-release"
GCS_BUCKET="drl-finance-results"

echo "======================================"
echo "Starting deployment of 1 experiments"
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

echo "Creating VM 1/1: trading-exp-00 for test..."
gcloud compute instances create trading-exp-00 \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator="type=$GPU_TYPE,count=1" \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=MIGRATE \
    --metadata-from-file=startup-script=cloud_deployment/startup_scripts/startup_test.sh \
    --tags=trading-experiment

if [ $? -eq 0 ]; then
    echo "Created trading-exp-00"
else
    echo "Failed to create trading-exp-00"
fi

sleep 3  # Small delay between VM creations

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
