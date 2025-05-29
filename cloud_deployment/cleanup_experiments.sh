#!/bin/bash

PROJECT_ID="your-project-id"

echo "Stopping and deleting all trading experiment VMs..."

# Get all VMs with the trading-experiment tag
VM_LIST=$(gcloud compute instances list --project=$PROJECT_ID --filter="tags:trading-experiment" --format="value(name,zone)")

if [ -z "$VM_LIST" ]; then
    echo "No trading experiment VMs found"
    exit 0
fi

while IFS=$'\t' read -r name zone; do
    echo "Deleting VM: $name in zone: $zone"
    gcloud compute instances delete $name --zone=$zone --project=$PROJECT_ID --quiet
done <<< "$VM_LIST"

echo "All trading experiment VMs deleted"
