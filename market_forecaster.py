import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score

def create_features(data, lags=5):
    """Creates a pandas DataFrame with lagged returns and the target variable."""
    df = pd.DataFrame(index=data.index)
    df['Today'] = data['Adj Close']
    df['Volume'] = data['Volume']
    for i in range(1, lags + 1):
        df[f'Lag_{i}'] = data['Adj Close'].pct_change(i) * 100
    df['Direction'] = np.sign(df['Today'].pct_change())
    df = df.dropna()
    df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
    return df

def main():
    # --- Step 1: Data Acquisition ---
    ticker = "SPY"
    start_date = "2001-01-01"
    end_date = "2018-12-31"
    data = yf.download(ticker, start=start_date, end=end_date)
    print("--- S&P 500 (SPY) Historical Data ---")
    print(data.tail())

    # --- Step 2: Feature Engineering ---
    features = create_features(data, lags=5)
    print("\n--- Features for Forecasting ---")
    print(features.head())

    # --- Step 3: Baseline Model Evaluation ---
    X = features[['Lag_1', 'Lag_2']]
    y = features['Direction']
    split_date = '2018-01-01'
    X_train = X[X.index < split_date]
    X_test = X[X.index >= split_date]
    y_train = y[y.index < split_date]
    y_test = y[y.index >= split_date]
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA()), ("SVM", SVC())]
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(f"Accuracy (Hit Rate): {accuracy:.4f}")
        print("Confusion Matrix:\n", conf_matrix)

    # --- Step 4: K-Fold Cross-Validation ---
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    print("\n--- K-Fold Cross-Validation (10 Folds on Training Data) ---")
    for name, model in models:
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        print(f"{name}: Mean Accuracy = {scores.mean():.4f} (Std Dev = {scores.std():.4f})")

    # --- Step 5: Hyperparameter Tuning with Grid Search ---
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
    grid.fit(X_train, y_train)
    
    print("\n--- Grid Search for Best SVM ---")
    print("Best parameters found: ", grid.best_params_)
    
    grid_predictions = grid.predict(X_test)
    accuracy = accuracy_score(y_test, grid_predictions)
    conf_matrix = confusion_matrix(y_test, grid_predictions)
    print(f"\nAccuracy of best SVM on test set: {accuracy:.4f}")
    print("Confusion Matrix:\n", conf_matrix)

if __name__ == "__main__":
    main()