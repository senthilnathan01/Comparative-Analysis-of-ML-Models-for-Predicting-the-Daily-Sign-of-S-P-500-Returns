# Comparative Analysis of ML-Models for Predicting the Daily Sign of S&P 500 Returns

## Project Overview

This project implements a machine learning model to forecast the daily direction of the S&P 500 index, using the SPDR S&P 500 ETF (SPY) as a proxy. The methodology is based on the concepts presented in Chapters 9, 11, and 16 of the book "Successful Algorithmic Trading" by Michael L. Halls-Moore.

The core idea is to use historical lagged returns as features to predict whether the next trading day's price will close higher or lower. The project systematically builds and evaluates several classification models, progressing from a baseline implementation to a more robust model using cross-validation and hyperparameter tuning.

---

## Methodology

1.  **Data Acquisition**: Historical daily price data for SPY is fetched using the `yfinance` library.
2.  **Feature Engineering**: Lagged daily returns are calculated to serve as predictive features. The target variable is a binary indicator representing the market's direction (+1 for up, -1 for down).
3.  **Baseline Model Evaluation**: The data is split into a training set and a held-out test set. Several baseline classification algorithms are trained and evaluated:
    *   Logistic Regression
    *   Linear Discriminant Analysis (LDA)
    *   Quadratic Discriminant Analysis (QDA)
    *   Support Vector Machine (SVM)
4.  **Robust Validation**: K-Fold Cross-Validation is applied to the training data to get a more reliable estimate of each model's performance and stability.
5.  **Hyperparameter Tuning**: `GridSearchCV` is used to find the optimal hyperparameters for the Support Vector Machine, which is then re-evaluated on the test set to measure its out-of-sample performance.

---

## Key Skills Showcased

*   **Financial Forecasting**: Application of machine learning to predict financial market movements.
*   **Time Series Analysis**: Feature engineering using lagged time series data.
*   **Machine Learning**: Proficiency with scikit-learn for model training, evaluation, and tuning.
*   **Quantitative Analysis**: Rigorous model validation using confusion matrices, hit rates, and cross-validation.
*   **Python for Finance**: Use of key scientific libraries including `pandas`, `numpy`, and `yfinance`.

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the analysis:**
    *   **Jupyter Notebook**: For an interactive experience with detailed explanations, open and run the cells in `market_forecaster.ipynb`.
        ```bash
        jupyter notebook market_forecaster.ipynb
        ```
    *   **Python Script**: To run the complete analysis from the command line, execute the script.
        ```bash
        python market_forecaster.py
        ```
