import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score
import os

from data_loader import fetch_and_save_data
from features import engineer_features

def train_and_evaluate_lgbm(df):
    # trains a LightGBM model with conservative hyperparameters
    # to avoid overfitting on noisy financial data
    # -> evaluates out-of-sample performance and feature importance
    print("Preparing data for LightGBM...")
    
    feature_cols = ['log_return', 'volatility_20d', 'z_score_20d']
    X = df[feature_cols]
    y = df['target_1d']
    
    # chronological split (80% train 20% out-of-sample test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training set: {len(X_train)} rows | Out-of-Sample set: {len(X_test)} rows\n")
    
    # create LGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # quant-conservative hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 15,       # kept small to prevent overfitting
        'max_depth': 4,         # shallow trees
        'feature_fraction': 0.8,
        'verbose': -1,
        'random_state': 42
    }
    
    print("Training LightGBM model...")
    # train model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # out-of-sample predictions
    y_proba_oos = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred_oos = (y_proba_oos > 0.5).astype(int)
    
    # calculate metrics
    acc = accuracy_score(y_test, y_pred_oos)
    auc = roc_auc_score(y_test, y_proba_oos)
    
    print("\n=== LightGBM Out-of-Sample Performance ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print("==========================================")
    
    # extract and print feature importance (gain)
    print("\n=== Feature Importance (Information Gain) ===")
    importance = model.feature_importance(importance_type='gain')
    for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True):
        print(f"{feat:20}: {imp:.2f}")
    print("=============================================\n")
    
    return model

if __name__ == "__main__":
    file_path = "data/SPY_daily.csv"
    
    if not os.path.exists(file_path):
        print("Data not found. Fetching now...")
        fetch_and_save_data("SPY", "2015-01-01", "2024-01-01")
        
    df_raw = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    df_features = engineer_features(df_raw)
    
    train_and_evaluate_lgbm(df_features)