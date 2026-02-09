# Hepatocellular Carcinoma (HCC) Survival Prediction 🏥

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)

A machine learning project for predicting survival outcomes in patients with Hepatocellular Carcinoma (HCC) using clinical and demographic data. This project demonstrates professional data science practices including modular code architecture, comprehensive testing, and reproducible workflows.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Team](#team)

## 🎯 Overview

Hepatocellular Carcinoma (HCC) is a common and aggressive type of liver cancer, often diagnosed in later stages due to subtle early symptoms. This project leverages machine learning techniques to predict the survival outcomes (Dies or Lives) of patients diagnosed with HCC.

**Key Objectives:**
- 📊 Analyze medical and demographic data from HCC patients
- 🤖 Build and compare multiple machine learning models
- 📈 Identify key factors influencing patient survival
- 🎯 Provide accurate prognostic predictions

## ✨ Features

- **Modular Code Architecture**: Clean, reusable Python modules for preprocessing, training, and evaluation
- **Multiple ML Models**: Random Forest, Decision Tree, and K-Nearest Neighbors classifiers
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports, and model comparison
- **Feature Importance Analysis**: Identify the most predictive clinical features
- **Reproducible Workflow**: Consistent random seeds and documented preprocessing steps
- **Professional Documentation**: Type hints, docstrings, and detailed README

## 📊 Dataset

The dataset (`hcc_dataset.csv`) contains real clinical data from **165 patients** diagnosed with HCC at the **Coimbra Hospital and University Center (CHUC)** in Portugal.

**Dataset Characteristics:**
- **Samples**: 165 patients
- **Features**: 49 clinical and demographic variables
- **Target**: Binary classification (Dies / Lives)
- **Source**: CHUC, Portugal

**Key Features Include:**
- Demographic information (age, gender)
- Medical history (symptoms, risk factors)
- Laboratory test results (AFP, bilirubin, albumin, etc.)
- Tumor characteristics

> **Note**: This dataset contains anonymized patient information. Please handle responsibly.

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/GuilhermeKotchergenko/data_exploration_hcc.git
cd data_exploration_hcc
```

2. **Create a virtual environment** (recommended)
```bash
# On Linux/Mac
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Option 1: Using the Modular Code

```python
# Import modules
from src.data_preprocessing import preprocess_pipeline
from src.model_training import train_all_models
from src.evaluation import evaluate_all_models

# Preprocess data
data = preprocess_pipeline(
    filepath='data/raw/hcc_dataset.csv',
    missing_strategy='median',
    test_size=0.2,
    random_state=42
)

# Train models
models = train_all_models(
    data['X_train'],
    data['y_train'],
    random_state=42
)

# Evaluate models
results = evaluate_all_models(
    models,
    data['X_test'],
    data['y_test'],
    save_dir='results'
)
```

### Option 2: Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open either:
# - notebooks/final_hcc.ipynb (original analysis)
# - notebooks/example_usage.ipynb (modular code demo)
```

## 📁 Project Structure

```
data_exploration_hcc/
├── data/
│   ├── raw/                    # Original dataset
│   │   └── hcc_dataset.csv
│   ├── processed/              # Preprocessed data (generated)
│   └── README.md
├── notebooks/
│   ├── final_hcc.ipynb         # Original comprehensive analysis
│   └── example_usage.ipynb     # Modular code usage example
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data loading and preprocessing
│   ├── model_training.py       # Model training functions
│   ├── evaluation.py           # Evaluation and visualization
│   └── utils.py                # Utility functions
├── models/                     # Saved trained models (generated)
├── results/                    # Evaluation results (generated)
│   ├── figures/                # Plots and visualizations
│   └── metrics/                # Performance metrics
├── tests/                      # Unit tests
├── docs/                       # Documentation
│   └── EIACD_Assignment2_2023_2024.pdf
├── .gitignore
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

## 💻 Usage

### Data Preprocessing

```python
from src.data_preprocessing import preprocess_pipeline

# Run complete preprocessing pipeline
data = preprocess_pipeline(
    filepath='data/raw/hcc_dataset.csv',
    missing_strategy='median',  # or 'mean', 'mode', 'drop'
    test_size=0.2,
    random_state=42
)
```

### Model Training

```python
from src.model_training import train_random_forest, train_decision_tree, train_knn

# Train individual models
rf_model = train_random_forest(X_train, y_train, hyperparameter_tuning=True)
dt_model = train_decision_tree(X_train, y_train)
knn_model = train_knn(X_train, y_train)

# Or train all models at once
from src.model_training import train_all_models
models = train_all_models(X_train, y_train, hyperparameter_tuning=False)
```

### Model Evaluation

```python
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

# Evaluate a single model
metrics = evaluate_model(model, X_test, y_test, model_name="Random Forest")

# Plot confusion matrix
plot_confusion_matrix(model, X_test, y_test, model_name="Random Forest")

# Plot ROC curve
plot_roc_curve(model, X_test, y_test, model_name="Random Forest")

# Compare all models
from src.evaluation import evaluate_all_models
results_df = evaluate_all_models(models, X_test, y_test, save_dir='results')
```

## 📈 Results

Our models achieve strong performance in predicting HCC patient survival outcomes:

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **0.85** | **0.84** | **0.85** | **0.84** | **0.89** |
| Decision Tree | 0.78 | 0.77 | 0.78 | 0.77 | 0.81 |
| K-Nearest Neighbors | 0.72 | 0.71 | 0.72 | 0.71 | 0.75 |

> **Note**: Results may vary slightly depending on the train-test split and hyperparameter settings.

### Key Findings

- 🏆 **Random Forest** achieves the best overall performance across all metrics
- 📊 **Feature Importance**: AFP (Alpha-fetoprotein), age, and bilirubin levels are among the most predictive features
- ⚖️ **Class Balance**: The dataset shows class imbalance, which is handled through stratified splitting

### Visualizations

The project generates comprehensive visualizations including:
- Confusion matrices for each model
- ROC curves with AUC scores
- Model performance comparison charts
- Feature importance plots

All visualizations are saved to `results/figures/` when running the evaluation pipeline.

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add some improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

This project was developed by FCUP (Faculty of Sciences, University of Porto) students:

- **Guilherme Kotchergenko Batista**
- **Rafael Arruda Costa**
- **Yan de Oliveira Christiano Coelho**

**Supervised by**: Professor Pedro Gabriel Dias Ferreira

## 📚 References

- Santos, M. et al. "A new cluster-based oversampling method for improving survival prediction of hepatocellular carcinoma patients." *Journal of Biomedical Informatics* 58 (2015): 49–59. [Link](https://www.sciencedirect.com/science/article/pii/S1532046415002063)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/user_guide/index.html)

## 🙏 Acknowledgments

- Coimbra Hospital and University Center (CHUC) for providing the dataset
- Professor Pedro Gabriel Dias Ferreira for supervision and guidance
- OpenAI ChatGPT for code review and assistance

---

**⭐ If you find this project useful, please consider giving it a star!**

For questions or feedback, please open an issue on GitHub.
