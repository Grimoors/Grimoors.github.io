---
layout: post
title: "Diving into NYC Taxi Trip Duration Prediction"
date: 2025-05-20 14:00:00 +0530
categories: machine-learning data-science mlops
---

# Diving into NYC Taxi Trip Duration Prediction

Hey everyone!

I'm excited to share my experience with the first homework assignment from the MLOps Zoomcamp 2025 cohort! This intensive program, led by DataTalks.Club, is helping me dive deep into the world of Machine Learning Operations, and this assignment was our first hands-on experience with a real-world prediction problem.

Recently, I tackled an interesting machine learning homework assignment focused on predicting the duration of taxi rides in New York City. It was a hands-on experience that took me through the fundamental steps of a data science project, from data acquisition and cleaning to model training and evaluation. Living here in Hyderabad, it was fascinating to work with data from such a vibrant city across the globe!

**What I Did: A Whirlwind Tour of Yellow Taxi Data**

My journey started with downloading the Yellow Taxi Trip Records for January and February 2023 from the NYC TLC website. It's amazing how much public data is available!

First, I loaded the January data using Python and the pandas library. The initial exploration involved figuring out the structure of the data – how many columns were there? It turned out to be **19** columns, each holding valuable information about the trips.

Next, I needed to engineer a crucial feature: the **trip duration**. By calculating the difference between pickup and dropoff timestamps and converting it to minutes, I got a clear picture of how long each ride lasted. I even calculated the standard deviation of these durations in January, which came out to be approximately **42.59**.

Data rarely comes perfectly clean, and this dataset was no exception. I encountered some outliers in the trip durations – rides that were either incredibly short or unusually long. To build a more robust model, I decided to filter out these extreme cases, keeping only the trips that lasted between 1 and 60 minutes. After this filtering, I was left with about **98%** of the original data. It's a good reminder that understanding data distributions and handling outliers is a critical step.

With the data cleaned, I moved on to feature engineering for my simple prediction model. I chose to focus on the pickup and dropoff location IDs (`PULocationID` and `DOLocationID`). To make these categorical features usable by a linear regression model, I applied **one-hot encoding**. This involved converting these ID columns into a sparse matrix where each unique location ID becomes a binary feature. Using scikit-learn's `DictVectorizer`, I transformed my data into this format. The resulting feature matrix had a dimensionality of **515** columns, highlighting the large number of unique pickup and dropoff locations.

Now came the exciting part: **training a model**. I opted for a simple linear regression model, aiming to predict the trip duration based on the one-hot encoded location features. Using the January data, I trained the model and then evaluated its performance on the same training data using the Root Mean Squared Error (RMSE). The RMSE on the training set was around **7.64** minutes, giving me an initial sense of the model's accuracy.

Finally, to see how well the model generalizes to new, unseen data, I used the February 2023 dataset as a **validation set**. I performed the same preprocessing steps (duration calculation, outlier removal, and one-hot encoding using the _same_ `DictVectorizer` fitted on the January data) on the February data. Evaluating the trained model on this validation set resulted in an RMSE of approximately **7.81** minutes. This validation RMSE gives a more realistic estimate of how the model would perform on future, unseen taxi trips.

**What I Learnt: Key Takeaways from the Taxi Trip Prediction Exercise**

This homework provided valuable insights into several key aspects of a machine learning workflow:

- **Data Acquisition and Exploration:** Understanding where to find relevant data and getting a feel for its structure (number of columns, data types) is the first crucial step.
- **Feature Engineering:** Creating meaningful features from raw data, like calculating trip duration from timestamps, can significantly impact model performance.
- **Data Cleaning and Preprocessing:** Identifying and handling outliers is essential for building robust models. Choosing appropriate filtering criteria requires understanding the data's characteristics.
- **Categorical Feature Encoding:** Techniques like one-hot encoding are necessary to transform categorical variables into a format that machine learning algorithms can understand. The dimensionality of the resulting feature space can be quite large depending on the number of unique categories.
- **Model Training and Evaluation:** Training a basic model like linear regression and evaluating its performance using metrics like RMSE provides a baseline understanding of the prediction task.
- **The Importance of Validation:** Evaluating a model on unseen data (the validation set) is critical for assessing its generalization ability and preventing overfitting. It highlights how well the model is likely to perform in the real world.
- **Using Existing Tools:** Libraries like pandas and scikit-learn provide powerful and efficient tools for data manipulation, feature engineering, and model building.

Overall, this homework was a great learning experience, solidifying my understanding of the fundamental steps involved in building a predictive model. Working with real-world data, even from a different city, provides invaluable practical experience.

Stay tuned for more explorations in the world of data!

---

## Step-by-Step To-Do: Predicting NYC Taxi Trip Duration

Here's a clear, step-by-step guide to reproduce the actions taken in this homework:

**1. Download the Data:**

- Visit the [NYC TLC Trip Record Data page](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Locate the "**Yellow** Taxi Trip Records" section.
- Download the `.parquet` files for:
  - January 2023
  - February 2023
- Save these files to a directory on your local machine.

**2. Analyze January Data (Q1):**

- Open a Python environment (e.g., Jupyter Notebook or a Python script).
- Import the pandas library:
  ```python
  import pandas as pd
  ```
- Load the January 2023 data:
  ```python
  df_jan = pd.read_parquet('path/to/your/yellow_tripdata_2023-01.parquet') # Replace 'path/to/your/'
  ```
- Get the number of columns:
  ```python
  num_columns = df_jan.shape[1]
  print(f"Number of columns: {num_columns}")
  ```
  _Identify this number for your Q1 answer._

**3. Compute Duration (Q2):**

- Ensure the datetime columns are in the correct format:
  ```python
  df_jan['tpep_pickup_datetime'] = pd.to_datetime(df_jan['tpep_pickup_datetime'])
  df_jan['tpep_dropoff_datetime'] = pd.to_datetime(df_jan['tpep_dropoff_datetime'])
  ```
- Calculate the duration:
  ```python
  df_jan['duration_td'] = df_jan['tpep_dropoff_datetime'] - df_jan['tpep_pickup_datetime']
  ```
- Convert the duration to minutes:
  ```python
  df_jan['duration'] = df_jan['duration_td'].dt.total_seconds() / 60
  ```
- Calculate the standard deviation of the `duration` column:
  ```python
  std_dev_duration = df_jan['duration'].std()
  print(f"Standard deviation of duration: {std_dev_duration}")
  ```
  _This is your answer for Q2._

**4. Drop Outliers (Q3):**

- Get the original number of records:
  ```python
  original_count = len(df_jan)
  ```
- Filter the DataFrame to keep rides between 1 and 60 minutes:
  ```python
  df_jan_filtered = df_jan[(df_jan['duration'] >= 1) & (df_jan['duration'] <= 60)].copy()
  ```
- Get the number of remaining records:
  ```python
  filtered_count = len(df_jan_filtered)
  ```
- Calculate the fraction of records left:
  ```python
  fraction_left = filtered_count / original_count
  print(f"Fraction of records left: {fraction_left}")
  ```
  _This is your answer for Q3._

**5. One-Hot Encode (Q4):**

- Import `DictVectorizer`:
  ```python
  from sklearn.feature_extraction import DictVectorizer
  ```
- Select the pickup and dropoff location ID columns (from the filtered dataframe):
  ```python
  categorical_cols = ['PULocationID', 'DOLocationID']
  df_features_train = df_jan_filtered[categorical_cols].copy()
  ```
- Convert the IDs to strings:
  ```python
  df_features_train['PULocationID'] = df_features_train['PULocationID'].astype(str)
  df_features_train['DOLocationID'] = df_features_train['DOLocationID'].astype(str)
  ```
- Convert the DataFrame to a list of dictionaries:
  ```python
  train_dicts = df_features_train.to_dict(orient='records')
  ```
- Initialize and fit the `DictVectorizer`:
  ```python
  dv = DictVectorizer()
  X_train = dv.fit_transform(train_dicts)
  ```
- Get the dimensionality of the feature matrix:
  ```python
  dimensionality = X_train.shape[1]
  print(f"Dimensionality of feature matrix: {dimensionality}")
  ```
  _This is your answer for Q4._

**6. Train a Model (Q5):**

- Import `LinearRegression` and `mean_squared_error`:
  ```python
  from sklearn.linear_model import LinearRegression
  from sklearn.metrics import mean_squared_error
  import numpy as np
  ```
- Prepare the target variable (from the filtered January data):
  ```python
  y_train = df_jan_filtered['duration'].values
  ```
- Initialize and train the linear regression model:
  ```python
  lr = LinearRegression()
  lr.fit(X_train, y_train)
  ```
- Make predictions on the training data:
  ```python
  y_pred_train = lr.predict(X_train)
  ```
- Calculate the RMSE:
  ```python
  rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
  print(f"RMSE on train: {rmse_train}")
  ```
  _This is your answer for Q5._

**7. Evaluate the Model (Q6):**

- Load the February 2023 data:
  ```python
  df_feb = pd.read_parquet('path/to/your/yellow_tripdata_2023-02.parquet') # Replace path
  ```
- Perform the same duration calculation (step 3) for `df_feb`:
  ```python
  df_feb['tpep_pickup_datetime'] = pd.to_datetime(df_feb['tpep_pickup_datetime'])
  df_feb['tpep_dropoff_datetime'] = pd.to_datetime(df_feb['tpep_dropoff_datetime'])
  df_feb['duration_td'] = df_feb['tpep_dropoff_datetime'] - df_feb['tpep_pickup_datetime']
  df_feb['duration'] = df_feb['duration_td'].dt.total_seconds() / 60
  ```
- Perform outlier removal (step 4) for `df_feb`, creating `df_feb_filtered`:
  ```python
  df_feb_filtered = df_feb[(df_feb['duration'] >= 1) & (df_feb['duration'] <= 60)].copy()
  ```
- Prepare the categorical features for February data:
  ```python
  df_features_val = df_feb_filtered[categorical_cols].copy() # Use same categorical_cols
  df_features_val['PULocationID'] = df_features_val['PULocationID'].astype(str)
  df_features_val['DOLocationID'] = df_features_val['DOLocationID'].astype(str)
  val_dicts = df_features_val.to_dict(orient='records')
  ```
- **Crucially, use the `dv` fitted on the January data to transform the February data:**
  ```python
  X_val = dv.transform(val_dicts) # Use the 'dv' from step 5
  ```
- Get the target variable for validation:
  ```python
  y_val = df_feb_filtered['duration'].values
  ```
- Make predictions on the validation data:
  ```python
  y_pred_val = lr.predict(X_val) # Use the 'lr' model from step 6
  ```
- Calculate the RMSE on the validation data:
  ```python
  rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
  print(f"RMSE on validation: {rmse_val}")
  ```
  _This is your answer for Q6._

By following these steps, you should be able to reproduce the results of the homework assignment. Good luck!

---

You can find my complete code implementation for this assignment in my GitHub repository: [MLOps Zoomcamp 2025 - Homework 1](https://github.com/Grimoors/mlopszoomcamp2025/tree/main/hw1)
