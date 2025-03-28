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

