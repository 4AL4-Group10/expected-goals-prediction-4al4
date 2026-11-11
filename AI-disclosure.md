## data_processing.py
1. 1 query used to understand how the StatsBomb data is organised and what a shot event contains
2. 3 queries used to understand what format the data needs to be in for model training and to figure out model features and inputs
3. 1 query used to help explore the data and find unusual or missing values
4. 2 queries used to help plan and understand how to break the preprocessing into stages
5. 1 query used to understand how to organise the script so teammates can import it for implementation and evaluation
6. 2 queries used to help with reading a match file and pulling out the shots and labels
7. 2 queries used to understand how to calculate shot distance and angle
8. 2 queries used to understand how to check whether a defender is between the shooter and goal
9. 2 queries used to help turn the freeze frame data into useful features
10. 1 query used to understand how to handle missing values without leaking information between splits
11. 1 query used to help add competition and gender information from the matches folder
12. 2 queries used to help with combining all the matches, removing bad shots, and saving the results
13. 2 queries used to understand one-hot encoding and how to keep the same columns across splits
14. 1 query used to understand data leakage and check which fields shouldn't be model inputs
15. 1 query used to help check that the output was correct

## models.py
1. 1 query used to help determine the best baselines to use in addition to the StatsBomb xG baseline
2. 2 queries used to help create class structures for the Logistic Regression and Constant Baseline classes before using Tutorial 3 techniques to implement fitting and prediction
3. 2 queries used to fix a few issues, including using correct operators (e.g. @ instead of *) and ensuring the standard deviation could not be 0
4. 2 queries used to re-implement the same cross-entropy formula for the constant baselines, ensuring a log(0) operation is not possible

## train.py
1. 1 query used to help ensure proper filtering in the data splitting part of the script and formatting based on split
2. 2 queries used to help break the training down into modules/parts
3. 3 queries used to help organise records for all models across search_hyperparameters and compare_baselines
4. 1 query used to help formulate the cross-entropy loss for the StatsBomb xG using the validation split
5. 2 queries used to help create print statements in some functions to track progress
6. 2 queries used to help with results saving and determining the best way to store data in tabular form in CSV and with joblib
7. 2 queries used to help determine and build a graph representing training cross-entropy over a given number of epochs

## evaluate.py
1. 3 queries used to summarise and interpret the pre-processing code and training code
2. 5 queries used to help recreate the test data set from the output artifacts of train.py and to explain what some lines of code do
3. 1 query used to retrieve the StatsBomb xG from the original processed dataset
4. 1 query used to help output calibration curves and loss results as a .csv
5. 3 queries used to get the loss formula for the Brier score and how to get ROC-AUC and the calibration curve with sklearn
6. 2 queries used to review the corresponding README entry for evaluation to ensure all conclusions made sense

## Introduction
1. 2 queries used to request wording and formatting feedback for revision

## Related Work
1. 1 query used to understand the bibliography structure of the code and how to use it
2. 2 queries used to request formatted citations for all of the sources used

## Dataset
1. 1 query used to understand what item 1 asks for in the Dataset section
2. 1 query used to understand how StatsBomb collects its data and the terms for using it
3. 2 queries used to help write a script to count the matches, competitions and shots in the dataset
4. 1 query used to understand the different shot outcomes and how to turn them into a goal / no goal label
5. 2 queries used to help find which shots should be removed and why, like shoot-outs and off-pitch locations
6. 1 query used to understand the missing values, like shots without freeze frames and missing true/false details
7. 2 queries used to help calculate goal rates by shot type, distance and defenders in the way
8. 1 query used to understand class imbalance and why it matters for this dataset
9. 1 query used to help create the goal rate table in LaTeX
10. 1 query used to check the dataset numbers against the preprocessing output

## Results and Evaluation
1. 1 query used to understand what item 5 asks for in the Results and Evaluation section
2. 1 query used to understand how the train, validation and test split should be done
3. 1 query used to understand whether cross-validation is needed for a dataset this size
4. 2 queries used to understand why accuracy is misleading for imbalanced data and which metrics to use instead
5. 2 queries used to understand what cross-entropy, Brier score and ROC-AUC mean and how to read them
6. 1 query used to understand how the majority vote and constant probability baselines work
7. 2 queries used to help interpret the test results and compare our model to StatsBomb's xG
8. 1 query used to understand what the calibration curves show and why the high xG bins are noisy
9. 1 query used to help create the results table in LaTeX

## Implementation
1. 4 queries used to provide the equations needed to explain the section and have them returned in LaTeX form
2. 1 query used to check whether any important equations were missed for a logistic regression model with SGD
3. 1 query used to decide whether to denote each row with the subscript i or keep the notation general
4. 1 query used to determine which details of mini-batch SGD are important to discuss
5. 1 query used to phrase the element-wise operation detail correctly
6. 1 query used to fix the phrasing of the hyperparameters shown, especially the sentence on sampling learning rates
7. 1 query used to format the figure and table with the necessary details

99 queries \* 4.32 g c02 emission = 427.68g c02 emission
