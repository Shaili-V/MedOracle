import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# LOADING, EXPLORING, AND INSPECTING THE DATASET (debugging help code commmented out):

# Load the dataset into a pandas DataFrame
data = pd.read_csv('data/SympScan.csv')
# Print the first 2 rows to verify the data is loaded correctly (.head() returns the first 5 rows by default)
   # print("First 2 rows of data:")
   # print(data.head(2))
# Print the shape of the dataset by showing the number of rows and columns
   # print("\nDataset shape (rows, columns):", data.shape)
# Get info about data types and non-null counts for each column
   # print("\nData summary:")
   # print(data.info())
# Check for missing values in each column
   # print("\nMissing values per column")
   # print(data.isnull().sum())
# Check distribution of target variable (disease column)
   # print("\nTarget variable distribution:")
   # print(data['diseases'].value_counts())
# Check distribution of a few symptom columns
   # print("\nDistribution of example symptom columns:")
   # example_symptoms = data.columns[1:6]
   # for symptom in example_symptoms:
       # print(f"{symptom} distribution:")
       # print(data[symptom].value_counts())
       # print()
# Check that all values are either 0 or 1 in the data cells
    # print("\nUnique values in each column:")
    # for column in data.columns:
        # unique_values = data[column].unique()
        # print(f"{column}: {unique_values}")

# DATA CLEANING AND PREPROCESSING:

# Split data into labels (y) and features (x) which is necessary for scikit-learn model training
y = data.iloc[:,0]
x = data.iloc[:,1:]
print(f"\nFeatures shape (x): {x.shape}")
print(f"Labels shape (y): {y.shape}")
# Split data into 80% training and 20% testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Confirm the split sizes
print(f"\nTraining set size: {x_train.shape}")
print(f"Testing set size: {x_test.shape}")

# BUILDING AND TRAINING THE MODEL:

# Create the Random Forest Classifier model instance with 100 trees and fixed random seed
rf = RandomForestClassifier(n_estimators = 50, max_depth =None, min_samples_split=5, min_samples_leaf=2, random_state=42)
# Train the model on training data
rf.fit(x_train, y_train)

# Predict on the test data
y_pred = rf.predict(x_test)
# Evaluate the model accuracy (test set)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest accuracy: {accuracy:.4f}")
# Evaluate the model training accuracy (train set)
y_train_pred = rf.predict(x_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_acc:.4f}") # If training accuracy is significantly higher than test accuracy, it may indicate overfitting.
# Print detailed classification report (precision, recall, f1-score)
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# TUNING THE MODEL: (grid search commented out, but results show at bottom of the section)

# Perform hyperparameter tuning using GridSearchCV
# param_grid = {
  #  'n_estimators': [50, 100],        # number of trees
  #  'max_depth': [None, 10, 20],      # max depth of each tree
  #  'min_samples_split': [2, 5],      # min samples to split a node
  #  'min_samples_leaf': [1, 2] }       # min samples at leaf node
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
# grid_search.fit(x_train, y_train)
# print("\nBest parameters:", grid_search.best_params_)
# print(f"Best cross-validated F1 score: {grid_search.best_score_:.4f}") 
    # Results: Best parameters: {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}. Best cross-validated F1 score: 0.8757 

# SAVING THE MODEL:

# Save the trained model to a file
joblib.dump(rf, 'random_forest_model.joblib')

