#!/bin/bash

set -e  # Exit script if any command fails
set -o pipefail

# 1. Clone repository
echo "Cloning repository..."
git lfs install
git clone https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds

# 2. Enter repository directory
cd unitree_sim_isaaclab_usds

# 3. Check if assets.zip exists and is greater than 1GB
if [ ! -f "assets.zip" ]; then
    echo "Error: assets.zip does not exist"
    exit 1
fi

filesize=$(stat -c%s "assets.zip")
if [ "$filesize" -le $((1024 * 1024 * 1024)) ]; then
    echo "Error: assets.zip is less than 1GB"
    exit 1
fi

echo "assets.zip check passed, size is $((filesize / 1024 / 1024)) MB"

# 4. Unzip assets.zip
echo "Unzipping assets.zip..."
unzip -q assets.zip

# 5. Move assets folder to parent directory
if [ -d "assets" ]; then
    echo "Moving assets to parent directory..."
    mv assets ../
else
    echo "Error: assets unzip failed or folder does not exist"
    exit 1
fi

# 6. Return to parent directory and delete original folder
cd ..
echo "Deleting unitree_sim_isaaclab_usds folder..."
rm -rf unitree_sim_isaaclab_usds

echo "All done!"
