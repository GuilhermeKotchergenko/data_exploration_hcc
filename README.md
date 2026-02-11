# Hepatocellular Carcinoma (HCC) Survival Prediction 🏥

A ML project for predicting survival outcomes in patients with Hepatocellular Carcinoma (HCC) using clinical and demographic data. This project demonstrates professional data science practices including modular code architecture, comprehensive testing, and reproducible workflows.

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

HCC is a common and aggressive type of liver cancer, often diagnosed in later stages due to subtle early symptoms. This project leverages machine learning techniques to predict the survival outcomes (Alive or Dead) of patients diagnosed with HCC.

**Key Objectives:**
- 📊 Analyze medical and demographic data from HCC patients
- 🤖 Build and compare ML models
- 📈 Identify key factors influencing patient survival
- 🎯 Provide accurate prognostic predictions

## ✨ Features

- **Modular Code Architecture**: Clean, reusable Python modules for preprocessing, training, and evaluation
- **Multiple ML Models**: Random Forest, Decision Tree, and K-NN classifiers
- **Comprehensive Evaluation**: Confusion matrices, ROC curves, classification reports, and model comparison
- **Feature Importance Analysis**: Identify the most predictive clinical features
- **Reproducible Workflow**: Consistent random seeds and documented preprocessing steps

## 📊 Dataset

The dataset (`hcc_dataset.csv`) contains real clinical data from **165 patients** diagnosed with HCC at the **Coimbra Hospital and University Center (CHUC)** in Portugal.

**Dataset Characteristics:**
- **Samples**: 165 patients
- **Features**: 49 clinical and demographic variables
- **Target**: Binary classification (Alive or Dead)
- **Source**: CHUC, Portugal

**Key Features Include:**
- Demographic information (age, gender)
- Medical history (symptoms, risk factors)
- Laboratory test results (AFP, bilirubin, albumin, etc.)
- Tumor characteristics

> **Note**: This dataset contains anonymized patient information.

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher

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

### Using Jupyter Notebooks

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

## 📈 Results and Analysis

### 1. Impact of Feature Engineering

We implemented:
- **Interaction Terms**: e.g., `Age * AFP` (capturing combined risk factors)
- **Log Transformations**: normalizing skewed distributions like `AFP`, `Creatinine`, `Ferritin`
- **Binning**: categorizing continuous variables like `Age` into risk groups

**Performance Comparison (F1-Score):**

| Model | Baseline F1 | Enhanced F1 | Improvement |
|-------|-------------|-------------|-------------|
| **Random Forest** | 0.5941 | **0.8149** | **+22.08%** |
| K-Nearest Neighbors | 0.6455 | 0.6973 | +5.18% |
| Decision Tree | **0.7599** | 0.6398 | -12.01% |

> **Key Takeaway**: Feature engineering was highly successful for the Random Forest model, transforming it from the worst-performing baseline model to the best-performing enhanced model.

### 2. Deep Dive: Model Behavior Analysis

**Why Random Forest Improved (+22%):**
- **Ensemble Power**: Random Forest aggregates predictions from many trees, allowing it to leverage the strong signals from engineered features (like interaction terms) while averaging out the noise.
- **Handling Complexity**: It effectively captured non-linear relationships introduced by log transforms and interactions without overfitting.

**Why Decision Tree Worsened (-12%):**
- **Overfitting**: Single decision trees are prone to overfitting. The addition of many new correlated features (e.g., predictors + their log versions) likely caused the tree to split on noise or redundant information.
- **Greedy Splitting**: The tree might have prioritized a new feature that worked well for a specific training subset but failed to generalize, unlike the robust consensus mechanism of the Random Forest.

### 3. Top Predictive Features

The analysis identified key biological drivers of survival prediction. The top features for the best-performing Random Forest model were primarily **engineered features**, validating our approach:

1.  **Age_AFP_Interaction**: Combined effect of patient age and Alpha-fetoprotein levels.
2.  **Log_AFP**: Log-transformed Alpha-fetoprotein (handling derived skewness).
3.  **AFP_Level**: Categorical risk bucket for AFP.
4.  **Age_Group**: Categorical risk bucket for Age.
5.  **Log_Creatinine**: Log-transformed kidney function metric.

### Visualizations

The project generates comprehensive visualizations including:
- Confusion matrices for each model
- ROC curves with AUC scores
- Feature importance plots (highlighting the dominance of engineered features)

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
