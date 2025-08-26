# House Price Prediction: Advanced Regression Techniques
### MSc Machine Learning Project

A comprehensive machine learning project implementing multiple regression models to predict house sale prices using the Ames Housing dataset from Kaggle.

## Overview

This project explores various regression techniques to predict residential property prices in Ames, Iowa, using 79 features covering property characteristics, location, condition, and construction details. The dataset contains sale prices ranging from $34,900 to $755,000 with a right-skewed distribution.

## Dataset

- **Source**: Kaggle "House Prices: Advanced Regression Techniques" competition
- **Features**: 79 variables covering:
  - Location and plot information
  - Property condition and quality metrics
  - Room characteristics and layout
  - Construction materials and methods
  - Time-related construction data
- **Target**: SalePrice (residential property sale prices)

## Data Preprocessing

### Feature Engineering
- **Quasi-constant feature removal**: Eliminated features with >99% identical values (PoolQC, PoolArea, Street, Utilities)
- **MSSubClass transformation**: Converted dwelling type codes to floor count categories
- **Log transformation**: Applied to LotArea to reduce outlier impact and improve distribution
- **Temporal features**: Converted dates to age differences relative to sale year
- **Seasonal clustering**: Used circular transformation and K-means clustering to group months by sale patterns
- **Feature scaling**: Applied standardization and min-max normalization as appropriate

### Feature Selection
- **Correlation filtering**: Removed redundant features with Pearson correlation >0.8
- **Sequential Feature Selection**: Forward selection using 5-fold cross-validation, identifying optimal 34-feature subset
- **Low-correlation removal**: Eliminated numerical features with <0.4 correlation to target

## Models Implemented

### 1. Linear Regression
- **Performance**: R² = 0.897 (test), 0.889 ± 0.026 (CV)
- **Key insight**: Logarithmic target transformation significantly improved performance and residual distribution

### 2. Polynomial Regression
- **Approach**: Used PCA (5 components) to manage dimensionality
- **Degrees tested**: 2-10
- **Best performance**: Degree 2 with R² = 0.867 (test)
- **Observation**: Higher degrees led to severe overfitting

### 3. Lasso Regression
- **Performance**: R² = 0.905 (test), 0.891 ± 0.024 (CV) - **Best overall**
- **Optimal λ**: 0.0008
- **Key features**: OverallQual showed strongest positive impact, temporal features showed negative correlation

### 4. Neural Networks
- **Architectures**: 1 and 2 hidden layers (10 neurons each)
- **Activation**: ReLU
- **Performance**: 
  - 1 layer: R² = 0.889 (test)
  - 2 layers: R² = 0.890 (test)
- **Finding**: Additional complexity provided minimal improvement

### 5. Gaussian Processes
- **Kernels**: Linear and RBF
- **Linear kernel**: R² = 0.897 (equivalent to Linear Regression)
- **RBF kernel**: R² = 0.889 with noise regularization (σ = 1)
- **Advantage**: Provides uncertainty estimates via confidence intervals

## Results Summary

| Model | Train R² | CV R² | Test R² | Target |
|-------|----------|--------|---------|---------|
| Linear Regression | 0.897 | 0.889 ± 0.026 | 0.897 | log(SalePrice) |
| Polynomial (deg 2) | 0.844 | 0.819 ± 0.135 | 0.867 | SalePrice |
| **Lasso Regression** | **0.902** | **0.891 ± 0.024** | **0.905** | **log(SalePrice)** |
| NN (1 layer) | 0.916 | 0.871 ± 0.052 | 0.889 | SalePrice |
| NN (2 layers) | 0.924 | 0.861 ± 0.087 | 0.890 | SalePrice |
| GP Linear | 0.897 | 0.889 ± 0.026 | 0.897 | log(SalePrice) |
| GP RBF | 0.892 | 0.855 ± 0.039 | 0.889 | SalePrice |

## Key Insights

1. **Logarithmic transformation** of the target variable consistently improved model performance by normalizing the skewed price distribution
2. **Feature selection** was crucial - 34 carefully selected features outperformed using all available features
3. **Lasso regression** achieved the best generalization performance through automatic feature selection and regularization
4. **Complex models** (deep neural networks, high-degree polynomials) didn't significantly outperform simpler approaches on this dataset
5. **OverallQual** emerged as the most predictive single feature across multiple models

## Files Structure
- `preprocess.ipynb` – Data preprocessing and feature engineering
- `train.ipynb` – Short feature selection, Model training and evaluation
- `train.csv` – Original Kaggle dataset
- `data_preprocessed.csv` – Cleaned dataset (generated)
- `plots/` - Directory of the saved plots
  
## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Run `preprocess.ipynb` to clean and transform the raw data
2. Run `train.ipynb` to perform feature selection and train models
3. Models are evaluated using 10-fold cross-validation and hold-out test set

## Author

**Konstantinos Vardakas**  
Student ID: 522  
Email: pcs0522@uoi.gr

## Course Information

Machine Learning - Exercise 1: Regression Problem

---

*This project demonstrates the application of various regression techniques and the importance of proper data preprocessing and feature engineering in machine learning pipelines.*
