# Quantitative ML Alpha Engine

**Evaluation of ML-Based Trading Signals with Python Research and C++ Inference**

This repository implements an end-to-end quantitative research pipeline. It evaluates machine learning models on financial time-series data to predict daily market direction, emphasizing statistically sound backtesting, feature engineering without lookahead bias, and low-latency production deployment.

## Tech Stack
* **Research & Data:** Python, Pandas, NumPy, yfinance, scikit-learn
* **Machine Learning:** LightGBM (Gradient Boosting)
* **Production Inference:** Modern C++, LightGBM C-API

## Project Architecture
1. **Feature Engineering:** Computes quantitative standard signals (log returns, rolling volatility, momentum Z-scores) ensuring strict adherence to chronological ordering to prevent data leakage.
2. **Walk-Forward Validation:** Evaluates models using rolling time-series cross-validation to simulate realistic, out-of-sample trading conditions.
3. **Model Training:** Trains a LightGBM classifier on historical ETF data (e.g., SPY) to predict next-day price direction.
4. **C++ Inference Engine:** Exports the trained Python model and executes it using the LightGBM C-API, demonstrating the transition from a Python research environment to a low-latency C++ production environment.

## Latency Benchmark
A benchmark evaluating the inference time per tick (predicting a single row of features). While Python adds GIL and API overhead, the native C++ implementation operates with significantly lower latency suitable for systematic trading systems.
* **Python API Latency:** ~X.XX µs per tick *(Run `src/benchmark.py` to see your local results)*
* **C++ Native Latency:** Significantly lower due to direct memory access and lack of interpreter overhead.

## How to Run (Python Research)
```bash
pip install -r requirements.txt
python src/data_loader.py    # Fetch ETF data
python src/features.py       # Compute quantitative signals
python src/model_lgbm.py     # Train & export the model
python src/benchmark.py      # Run latency benchmark