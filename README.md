
# Credit Default Prediction

This repository walks through the full process of:
- Exploring a real-world credit dataset
- Preprocessing data ethically and legally
- Building interpretable and high-performing models (Logistic Regression & XGBoost)
- Translating output probabilities into **FICO-like credit scores**

---

## 📁 Project Structure

### Part 1: Logistic Regression
- Data loading from Excel
- Exploratory data analysis:
  - Summary statistics
  - Correlation matrix
  - Distribution plots
- Preprocessing with `ColumnTransformer`:
  - Numerical: imputation + scaling
  - Categorical: imputation + one-hot encoding
- Logistic regression training with cross-validation
- ROC curve and feature importance (via Random Forest)
- Model saved as `Models/logistic_regression_model.pkl`

---

### Part 2: XGBoost Modeling
- Uses `xgboost.DMatrix` for training
- Extensive grid search using `xgb.cv` for hyperparameter tuning
- Best model trained and evaluated on:
  - Accuracy
  - AUC
  - Precision, Recall, F1-Score
- Feature importance plotted from tree weights
- Model saved as `Models/xgboost_model.model`

---

### Part 3: FICO Score Approximation
- A post-processing function converts predicted probabilities into **FICO-style credit scores** (range 300–850):

```python
def calculate_fico_scores(pred, A=850, B=300):
    odds = pred / (1 - pred)
    return np.clip(A - B * np.log10(odds), B, A)
```

This allows model outputs to be interpreted within industry-standard scoring formats.

---

## 💾 Dataset

- **Source**: UCI Machine Learning Repository  
  **Name**: *Default of Credit Card Clients Dataset*  
  **File**: `default of credit card clients.xls`

- **Size**: 30,000 entries  
- **Target**: `default payment next month` (0 or 1)

---

## 🧠 Model Performance

| Model               | Accuracy | ROC AUC | Precision | Recall | F1 Score |
|--------------------|----------|---------|-----------|--------|----------|
| Logistic Regression| 0.8060   | 0.7161  | 0.6983    | 0.2163 | 0.3303   |
| XGBoost            | 0.8010   | 0.7622  | 0.7806    | 0.1394 | 0.2366   |

---

## 🧪 Requirements

- `scikit-learn`
- `xgboost`
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `optuna` (for advanced tuning)
- `openpyxl` (for reading `.xls` files)

---

## 💡 Usage

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Jupyter Notebook and execute each cell:
   - Preprocessing
   - Training
   - Evaluation
   - FICO scoring

---

## 📜 License

Educational use only. Dataset used is in the public domain via [UCI Repository](https://archive.ics.uci.edu/).

---

## 🏁 Final Thoughts

- XGBoost performed better in ROC AUC
- Logistic Regression was more interpretable
- FICO-style scoring allows industry-aligned deployment

Both models are production-ready and saved to disk.
