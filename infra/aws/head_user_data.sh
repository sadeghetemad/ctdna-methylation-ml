#!/bin/bash
# Head node setup: Java, Nextflow, git and Miniforge (for the pipeline's conda env).
# Progress: /var/log/head-setup.log
exec > /var/log/head-setup.log 2>&1
set -ex

dnf install -y git java-17-amazon-corretto-headless tmux

curl -fsSL https://get.nextflow.io | bash
mv nextflow /usr/local/bin/nextflow
chmod 755 /usr/local/bin/nextflow

curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p /opt/miniforge
chmod -R a+rX /opt/miniforge
ln -sf /opt/miniforge/bin/conda /usr/local/bin/conda
echo 'source /opt/miniforge/etc/profile.d/conda.sh' > /etc/profile.d/conda.sh

mkdir -p /data
chmod 777 /data
touch /var/log/head-setup.done
