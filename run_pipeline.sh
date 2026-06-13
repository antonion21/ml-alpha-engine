#!/bin/bash

# stops script on error
set -e 

echo "======================================================"
echo "   🚀 ML Alpha Engine - End-to-End Pipeline"
echo "======================================================"

echo -e "\n[1/5] Fetching Market Data (SPY)..."
python3 src/data_loader.py

echo -e "\n[2/5] Engineering Quantitative Features..."
python3 src/features.py

echo -e "\n[3/5] Training LightGBM Classifier..."
python3 src/model_lgbm.py

echo -e "\n[4/5] Generating 3D Hyperparameter Surface..."
python3 src/plot_surface.py

echo -e "\n[5/5] Executing Native C++ Inference Engine..."
./build/inference

echo "======================================================"
echo "   ✅ Pipeline Execution Completed Successfully!"
echo "======================================================"