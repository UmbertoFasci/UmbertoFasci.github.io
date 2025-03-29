---
layout: post
title:  Simulated Regional Profit Analysis & Recommendations - Oil and Mining Application
description: A modeling procedure which is designed to predict oil reserve volume, and provide a bootstrapped profit analysis on a regional basis, allowing for optimal oil well placement. 
date:   2025-03-27 15:01:35 +0300
image:  '/images/bigoil.jpg'
tags:   [Predictive Modeling, Business]
---
# Introduction
The primary goal of this project is to build a machine learning model capable of accurately predicting the volume of oil reserves for new wells in each region. This model will help select the top-performing oil wells based on predicted reserves and identify the region with the highest profit margin. The model’s output will guide the decision on where to drill, considering both profitability and financial risk.

For the purpose of understanding the main structure of the following analysis, I have taken a modular approach to each of the sections and subsections. Creating functions to perform the modeling, analysis all with defined output structure in order to further utilize the resulting information.

## Data Ingestion and Initial Exploration

All duplicates expressed throughout each of the datasets are only duplicated across the id subset. Looking closely none of the features or products throughout this duplicate range are duplicated themselves. This suggests that these are additional measuremets taken at the same well id.

In the end, in order to avoid any potential issues and considering the negligable amount of this data, these duplicated data were dropped.

_This procedure is further detailed in the following code block..._ 

```python
file_paths = ["../datasets/geo_data_0.csv", "../datasets/geo_data_1.csv", "../datasets/geo_data_2.csv"]

geodata = []
duplicates_df = pd.DataFrame()  # Create empty DataFrame for duplicates

for path in file_paths:
    # Load the data
    data = pd.read_csv(path)
    
    # Find duplicates
    duplicates = data[data.duplicated(subset='id', keep=False)].copy()
    if not duplicates.empty:
        # Add a column to indicate which file the duplicates came from
        duplicates['source_file'] = path.split('/')[-1]
        duplicates_df = pd.concat([duplicates_df, duplicates], ignore_index=True)
    
    # Remove duplicates based on the 'id' column and add to the list
    geodata.append(data.drop_duplicates(subset='id'))

# Sort duplicates_df by ID to group duplicates together
duplicates_df = duplicates_df.sort_values(['id']).reset_index(drop=True)

# Unpack the data into separate variables if needed
geo_data_0, geo_data_1, geo_data_2 = geodata
```
## Regional Feature Distributions

The following section describes the data distributions of each of the regions. 

### Region 1 Feature Distributions

- `f0`: Shows a multimodal distribution with several peaks, roughly symmetric but with clear separations between clusters
- `f1`: Similar to f0, displays multimodal behavior with 3-4 distinct peaks and valleys
- `f2`: Appears more unimodal and normally distributed, centered around 0-2 with some slight right skew

### Region 2 Feature Distributions:

- `f0`: Bimodal distribution with two prominent peaks, fairly symmetric around 0
- `f1`: Single normal/Gaussian distribution, unimodal and symmetric
- `f2`: Appears to be a discrete-looking distribution with regularly spaced spikes, though noted as continuous. This could indicate heavily quantized or binned data.

### Region 3 Feature Distributions:

- `f0`: Single peaked, normal distribution with slight asymmetry
- `f1`: Normal distribution, very symmetric and unimodal
- `f2`: Normal distribution with slight heavy tails, symmetric and unimodal

# Base Model

For the purposes of this analysis, **Linear Regression** is the chosen algorithm to model the regional `product` target. Additionally, `RMSE` is utilized as the main metric to measure model performance. _Additional metrics are utilized in tendem to inform model performance._ Model parameters are collected and stored into the `model_params` dictionary in occordance to a custom output structure workflow. Working with stored data in this fashion allows for much more versatility in future functionalities such as a DataFrame showcasing the results of the model.

```python
def train_and_evaluate_region(data, region_name):
    # Separate features and target
    features = data[['f0', 'f1', 'f2']]
    target = data['product']
    
    # Split data into training and validation sets (75:25)
    features_train, features_val, target_train, target_val = train_test_split(features, target, test_size=0.25, random_state=12345)
    
    # Initialize and train model
    model = LinearRegression()
    model.fit(features_train, target_train)
    
    # Make predictions on validation set
    target_pred = model.predict(features_val)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(target_val, target_pred))
    r2 = r2_score(target_val, target_pred)
    
    # Save validation results
    validation_results = pd.DataFrame({
        'Actual': target_val,
        'Predicted': target_pred,
        'Error': target_val - target_pred
    })
    
    # Get feature coefficients
    feature_coefficients = dict(zip(features.columns, model.coef_))
    
    model_params = {
        'param_fit_intercept': model.fit_intercept,
        'param_copy_X': model.copy_X,
        'param_positive': model.positive,
        'param_n_features_in': model.n_features_in_
    }
    
    return {
        'region_name': region_name,
        'model': model,
        'rmse': rmse,
        'r2': r2,
        'avg_predicted': np.mean(target_pred),
        'avg_actual': np.mean(target_val),
        'validation_results': validation_results,
        'feature_coefficients': feature_coefficients,
        'intercept': model.intercept_,
        'parameters': model_params
    }
```

## Base Model Results

| Region | RMSE | R2 Score | Avg Predicted Volume | Avg Actual Volume | f0 | f1 | f2 | Intercept |
|--------|------|----------|----------------------|-------------------|----|----|----|-----------|
|   1    | 37.85 | 0.27 | 92.78 |  92.15 | 3.78 | -13.89 | 6.63 | 77.63 |
|   2    | 0.89  | 0.99 | 69.17 | 69.18 | -0.14 | -0.022 | 26.95 | 1.65 |
|   3    | 40.07 | 0.19 | 94.86 |  	94.78 | 0.052 | -0.061 |  	5.77 | 80.61 |

Region 2 shows excellent performance: - Very high R² (0.999) indicating the model explains nearly all variance in the data - Very low RMSE (0.89) showing high prediction accuracy - The scatter plot shows points tightly clustered along the perfect prediction line - Error distribution is narrow and normally distributed with small standard deviation - Predicted vs actual volumes are nearly identical (69.18 vs 69.19) - The highly quantized data exhibited by both the f2 feature and the target potentially hold strong influence on the results of this model.

Regions 1 and 3 show poor performance: - Low R² values (0.27 and 0.20 respectively) indicating the model explains very little of the variance - High RMSE values (37.85 and 40.08) showing large prediction errors - Scatter plots show wide dispersion from the perfect prediction line - Error distributions are much wider with larger standard deviations - While average predicted volumes are close to actuals, individual predictions vary greatly

Reviewing the assumptions made by Linear Regression modeling there is an assumption where the errors are normally distributed this is why I have decided to include these in the model performance analysis.