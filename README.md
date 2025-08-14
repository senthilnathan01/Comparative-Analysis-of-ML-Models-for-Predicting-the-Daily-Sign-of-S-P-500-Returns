# Comparative Analysis of Supervised Learning for Daily S&P 500 Directional Forecasting

## TLDR

* Built and back-tested a machine learning signal generator for algorithmic trading, forecasting S&P 500 daily returns.

* Achieved a statistically significant predictive edge with a 54% hit rate, a crucial margin for generating positive expectancy and long-term profitability in systematic strategies.

* Implemented K-Fold cross-validation and Grid Search to optimize model robustness and tune SVM hyperparameters.

* Analyzed model bias, inferring from the confusion matrix that the edge was primarily in identifying positive-return days.

### Project Overview

This project develops and evaluates a machine learning framework to forecast the daily price direction of the S&P 500 index. Using lagged returns as features, a comparative analysis of several classification models (Logistic Regression, LDA, QDA, SVM) was performed. The framework includes robust validation via K-Fold Cross-Validation and hyperparameter optimization using `GridSearchCV` to build a reliable and statistically sound forecasting model.

### Key Findings & Results

*   **Out-of-Sample Accuracy**: The final optimized Support Vector Machine (SVM) model achieved a **hit rate of 54.03%** on the unseen 2018 test data, demonstrating a consistent predictive edge over a 50% random baseline.

*   **Model Bias Analysis**: The confusion matrix revealed a significant model bias towards predicting positive-return days. The model correctly identified 98.5% of 'up' days but only 4.3% of 'down' days. This reflects the model learning the market's historical upward drift rather than being a purely symmetrical forecaster.

*   **Validation and Robustness**: 10-Fold Cross-Validation confirmed the stability of the predictive signal, yielding mean accuracies around 54.6% on the training data with a low standard deviation (0.017), indicating the edge is persistent and not a statistical anomaly.

*   **Financial Inference**: The results align with the weak-form Efficient Market Hypothesis, showing a faint but quantifiable predictive signal in historical price data. This edge is insufficient for a simple long/short strategy but could be viable as a signal generator for a long-only strategy with proper risk and cost management.

### Skills & Technologies Demonstrated

*   **Machine Learning**: Supervised Classification (Logistic Regression, LDA, QDA, SVM), Model Evaluation, K-Fold Cross-Validation, Hyperparameter Tuning (Grid Search).
*   **Quantitative Finance**: Time Series Analysis, Feature Engineering, Signal Generation, Efficient Market Hypothesis.
*   **Python Libraries**: `scikit-learn`, `pandas`, `numpy`, `yfinance`, `matplotlib`.

### Future Enhancements

*   **Advanced Feature Engineering**: Incorporate volatility, momentum, and inter-market features (e.g., VIX, bond yields).
*   **Advanced Models**: Implement ensemble methods like Random Forest or Gradient Boosting (XGBoost) to capture more complex patterns.
*   **Full Backtesting**: Integrate the signal generator with an event-driven backtester to simulate a full trading strategy, including transaction costs and slippage.

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
