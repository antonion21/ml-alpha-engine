import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_auc_score
import os

from features import engineer_features

def generate_surface_plot(df):
    print("=== Starting Grid Search for 3D Surface Plot ===\n")
    
    feature_cols = ['log_return', 'volatility_20d', 'z_score_20d']
    X = df[feature_cols]
    y = df['target_1d']
    
    # chronological split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # our grid (3x3 = 9 models)
    learning_rates = [0.01, 0.05, 0.1]
    num_leaves_list = [7, 15, 31]
    
    # meshgrid for 3D plot creation
    LR, NL = np.meshgrid(learning_rates, num_leaves_list)
    performance = np.zeros(LR.shape)
    
    # training models and measuring out-of-sample AUC
    for i in range(LR.shape[0]):
        for j in range(LR.shape[1]):
            lr = LR[i, j]
            nl = int(NL[i, j])
            
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': lr,
                'num_leaves': nl,
                'verbose': -1,
                'random_state': 42
            }
            
            model = lgb.train(params, train_data, num_boost_round=50)
            y_proba = model.predict(X_test)
            auc = roc_auc_score(y_test, y_proba)
            performance[i, j] = auc
            
            print(f"Trained: LR={lr:<4} | Leaves={nl:<2} --> Out-of-Sample AUC: {auc:.4f}")

    # create 3d surface plot
    print("\nGenerating 3D plot...")
    os.makedirs("plots", exist_ok=True)
    
    # styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # draw surfaces
    surf = ax.plot_surface(
        LR, NL, performance, 
        cmap='viridis', 
        edgecolor='black',
        linewidth=0.5,
        alpha=0.9
    )
    
    # Label axes
    ax.set_xlabel('\nLearning Rate', fontsize=10)
    ax.set_ylabel('\nNumber of Leaves', fontsize=10)
    ax.set_zlabel('\nOut-of-Sample AUC', fontsize=10)
    ax.set_title('LightGBM Hyperparameter Sensitivity Surface\n(Stability Diagnostic)', fontsize=12, pad=20)
    
    fig.colorbar(surf, shrink=0.5, aspect=10, label='AUC Score', pad=0.1)
    
    plot_path = "plots/hyperparameter_surface.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 3D plot successfully saved to: {plot_path}")

if __name__ == "__main__":
    file_path = "data/SPY_daily.csv"
    if not os.path.exists(file_path):
        print("Please run data_loader.py first!")
    else:
        df_raw = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df_features = engineer_features(df_raw)
        generate_surface_plot(df_features)