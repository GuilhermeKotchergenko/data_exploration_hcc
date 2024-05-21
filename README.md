# Hepatocellular Carcinoma (HCC) Survival Prediction

This project aims to enhance understanding of data visualization, preprocessing, and the prediction of survival outcomes (Dies or Lives) for patients with Hepatocellular Carcinoma (HCC) using various machine learning models. The dataset includes medical and demographic information collected from patients at the Coimbra Hospital and University Center (CHUC) in Portugal, containing real clinical data of individuals diagnosed with HCC.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Bibliography](#bibliography)
- [Team](#project-supervision-and-contribution)

## Introduction

Hepatocellular Carcinoma (HCC) is a common and aggressive type of liver cancer, often diagnosed in later stages due to subtle early symptoms. This project leverages machine learning techniques to predict the survival outcomes (Dies or Lives) of patients diagnosed with HCC. By analyzing a comprehensive dataset that includes both medical and demographic information, the project aims to provide insights into the factors that influence survival and improve prognostic accuracy for HCC patients.

## Dataset

The dataset used in this project is `hcc_dataset.csv`, which contains the following information:
- Medical history and demographic data of patients
- Various medical test results
- Survival outcome (Dies or Lives)

## Dependencies

The project requires the following Python libraries, all of which are included in the ANACONDA package:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the ANACONDA package at [Anaconda Download page](https://www.anaconda.com/products/distribution) and the appropriate installer for your operating system.

You can also install the dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Project Structure

- `final_hcc.ipynb`: Jupyter notebook containing the code for the project.
- `hcc_dataset.csv`: Dataset used for the project.

## Usage

1. **Clone the repository**:

```bash
   git clone https://github.com/your-username/HCC-Survival-Prediction.git
   cd HCC-Survival-Prediction
```

2. **Run the Jupyter Notebook**:

   Open `Checkpoint1.ipynb` in Jupyter Notebook or Jupyter Lab and execute the cells to run the project.

## Results

The project evaluates three machine learning models:
- Random Forest
- Decision Tree
- K-Nearest Neighbors (KNN)

The models are evaluated based on the following metrics:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve and AUC

### Model Comparison

A summary table and bar chart compare the performance of the three models based on the evaluation metrics mentioned above.

## Bibliography

- Professor Pedro Gabriel Dias Ferreira’s EIACD notebook
- Scikit-learn.org user guide
- Pandas.pydata.org user guide
- OpenAI ChatGPT – code supervision

## Project Supervision and Contribution

This project was supervised by:

- Professor Pedro Gabriel Dias Ferreira

and developed by FCUP students:

- Guilherme Kotchergenko Batista

- Rafael Arruda Costa

- Yan de Oliveira Christiano Coelho
