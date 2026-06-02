import time
import lightgbm as lgb
import numpy as np
import os

def run_benchmark():
    print("=== Latency Benchmark: Python Inference ===\n")
    
    model_path = "models/lgbm_model.txt"
    if not os.path.exists(model_path):
        print(f"[Fehler] Modell nicht gefunden in {model_path}.")
        return

    model = lgb.Booster(model_file=model_path)
    print("[OK] LightGBM Modell geladen.")

    # dummy features (log_return, volatility_20d, z_score_20d)
    dummy_features = np.array([[0.0015, 0.0120, 1.500]])

    # warmup (important for accurate measurements)
    for _ in range(10):
        model.predict(dummy_features)

    # benchmark
    n_runs = 1000
    start_time = time.perf_counter()
    
    for _ in range(n_runs):
        model.predict(dummy_features)
        
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / n_runs
    avg_time_us = avg_time_ms * 1000
    
    print(f"Anzahl Vorhersagen: {n_runs}")
    print(f"Gesamtzeit:         {total_time_ms:.2f} ms")
    print(f"Ø Latenz pro Tick:  {avg_time_us:.2f} µs")
    print("\n[Info] C++ Inference (Commit 7) läuft in der Praxis je nach")
    print("Systemarchitektur und Memory-Layout nochmals deutlich schneller,")
    print("da der Python-GIL und API-Overhead komplett entfallen.")
    print("===========================================\n")

if __name__ == "__main__":
    run_benchmark()