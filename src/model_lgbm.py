import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import os

from data_loader import fetch_and_save_data
from features import engineer_features

def train_and_export_lgbm(df, model_dir="models"):
    # trains a LGBM model with conservative hyperparameters
    # and exports it for C++ low-latency inference
    print("Preparing data for LightGBM...")
    
    feature_cols = ['log_return', 'volatility_20d', 'z_score_20d']
    X = df[feature_cols]
    y = df['target_1d']
    
    # chronological split (80% train 20% out-of-sample test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # evaluate out-of-sample
    y_proba_oos = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, y_proba_oos)
    print(f"\nOut-of-Sample AUC: {auc:.4f}")
    
    # export model for C++ inference
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lgbm_model.txt")
    
    # LGBM saves the tree structure in a format readable by its C API
    model.save_model(model_path, num_iteration=model.best_iteration)
    print(f"Model successfully exported to: {model_path}")
    print("Ready for C++ inference.")
    
    return model

if __name__ == "__main__":
    file_path = "data/SPY_daily.csv"
    
    if not os.path.exists(file_path):
        print("Data not found. Fetching now...")
        fetch_and_save_data("SPY", "2015-01-01", "2024-01-01")
        
    df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df_features = engineer_features(df_raw)
    
    train_and_export_lgbm(df_features)