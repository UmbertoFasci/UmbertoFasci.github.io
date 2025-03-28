---
layout: post
title:  Simulated Regional Profit Analysis & Recommendations - Oil and Mining Application
description: Procedure outlining 
date:   2025-03-27 15:01:35 +0300
image:  '/images/bigoil.jpg'
tags:   [Predictive Modeling, Business]
---
# Introduction
The primary goal of this project is to build a machine learning model capable of accurately predicting the volume of oil reserves for new wells in each region. This model will help select the top-performing oil wells based on predicted reserves and identify the region with the highest profit margin. The model’s output will guide the decision on where to drill, considering both profitability and financial risk.

For the purpose of understanding the main structure of the following analysis, I have taken a modular approach to each of the sections and subsections. Creating functions to perform the modeling, analysis all with defined output structure in order to further utilize the resulting information.

## Data Ingestion and Initial Exploration

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
