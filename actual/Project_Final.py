import pandas as pd
from tqdm.auto import tqdm

# Load the CSV file into a DataFrame
file_path = 'DataSet.csv'
data = pd.read_csv(file_path, encoding='UTF-8-SIG')

# Display the head of the DataFrame
data_head = data.head()

# Show the head of the DataFrame
print(data_head)

from tqdm.auto import tqdm

# Check for missing values
missing_values = data.isnull().sum()

# Check for duplicate rows
duplicate_rows = data.duplicated().sum()

# Check for any obvious issues with data types
data_types = data.dtypes

# Summary statistics to identify any outliers or anomalies
summary_statistics = data.describe()

# Display the findings
print('Missing Values:\n', missing_values)
print('\nDuplicate Rows:', duplicate_rows)
print('\nData Types:\n', data_types)
print('\nSummary Statistics:\n', summary_statistics)

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Select columns to scale, typically numerical columns
numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

# Apply scaling to the numerical columns
data_scaled = data.copy()
data_scaled[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Display the head of the scaled DataFrame
data_scaled_head = data_scaled.head()

# Show the head of the scaled DataFrame
print(data_scaled_head)

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Apply PCA for feature extazraction
pca = PCA(n_components=0.95) # Keep 95% of variance
principal_components = pca.fit_transform(data_scaled[numerical_cols])

# Convert to DataFrame for easier handling
pca_df = pd.DataFrame(data=principal_components)

# Apply SelectKBest for feature selection
selector = SelectKBest(f_classif, k='all')
selector.fit(data_scaled[numerical_cols], data_scaled['TARGET'])

# Get the scores for each feature
feature_scores = selector.scores_

# Convert to DataFrame for easier handling
feature_scores_df = pd.DataFrame({'Feature': numerical_cols, 'Score': feature_scores})

# Sort the DataFrame by the scores in descending order
feature_scores_df = feature_scores_df.sort_values(by='Score', ascending=False)

# Display the PCA DataFrame head and the feature scores
print('PCA DataFrame (head):')
print(pca_df.head())
print('\nFeature Scores:')
print(feature_scores_df)

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load your dataset
data_scaled = pd.read_csv('DataSet.csv')

# Define your features (X) and target variable (y)
X = data_scaled.drop('TARGET', axis=1)
y = (data_scaled['TARGET'] > 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100)

# Fit the Random Forest classifier on the training data
rf_clf.fit(X_train, y_train)

# Instantiate the SVM classifier
svm_clf = SVC(kernel='linear')  # You can choose a different kernel based on your needs

# Fit the SVM classifier on the training data
svm_clf.fit(X_train, y_train)

# Now, you can use both trained models for predictions or evaluation on the test set


# Assuming 'data_scaled.csv' is your dataset file
data_scaled = pd.read_csv('DataSet.csv')

# Define your features (X) and target variable (y)
X = data_scaled.drop('TARGET', axis=1)
y = (data_scaled['TARGET'] > 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100)

# Fit the Random Forest classifier on the training data
rf_clf.fit(X_train, y_train)

# Now, you can use the trained model for predictions or evaluation on the test set

# Convert the target variable back to categorical
y = (data_scaled['TARGET'] > 0).astype(int)

# Split the data into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retrain the Random Forest classifier
rf_clf.fit(X_train, y_train)
# Retrain the SVM classifier
svm_clf.fit(X_train, y_train)

# Make predictions with both classifiers
rf_predictions = rf_clf.predict(X_test)
svm_predictions = svm_clf.predict(X_test)

# Combine predictions - here we will simply average them
hybrid_predictions = (rf_predictions + svm_predictions) / 2

# Evaluate the hybrid model
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions.round())

# Print the accuracy of the hybrid model
print('Hybrid Model Accuracy:', hybrid_accuracy)


# Convert the target variable back to categorical
X = data_scaled.drop('TARGET', axis=1)
y = (data_scaled['TARGET'] > 0).astype(int)

# Split the data into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Retrain the Random Forest classifier
rf_clf.fit(X_train, y_train)
# Retrain the SVM classifier
svm_clf.fit(X_train, y_train)

# Make predictions with both classifiers
rf_predictions = rf_clf.predict(X_test)
svm_predictions = svm_clf.predict(X_test)

# Combine predictions - here we will simply average them
hybrid_predictions = (rf_predictions + svm_predictions) / 2

# Evaluate the hybrid model
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions.round())

# Print the accuracy of the hybrid model
print('Hybrid Model Accuracy:', hybrid_accuracy)

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

class HybridModel(BaseEstimator):
    def __init__(self):
        self.rf_clf = RandomForestClassifier(random_state=42)
        self.svm_clf = SVC(probability=True, random_state=42)

    def fit(self, X, y):
        self.rf_clf.fit(X, y)
        self.svm_clf.fit(X, y)
        return self

    def predict(self, X):
        rf_predictions = self.rf_clf.predict_proba(X)[:, 1]
        svm_predictions = self.svm_clf.decision_function(X)
        hybrid_predictions = (rf_predictions + svm_predictions) / 2
        return (hybrid_predictions > 0.5).astype(int)

# Initialize the hybrid model
hybrid_model = HybridModel()

# Perform cross-validation
scores = cross_val_score(hybrid_model, X, y, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of the cross-validation scores
mean_score = scores.mean()
std_dev_score = scores.std()

# Display the results
print('Cross-validated scores:', scores)
print('Mean accuracy:', mean_score)
print('Standard deviation:', std_dev_score)

# Fit the hybrid model on the entire dataset

hybrid_model.fit(X, y)

# Predict on the test set
y_pred = hybrid_model.predict(X_test)

# Calculate performance metrics
from sklearn.metrics import classification_report
performance_metrics = classification_report(y_test, y_pred, output_dict=True)

# Convert performance metrics to a DataFrame for better visualization
import pandas as pd
performance_df = pd.DataFrame(performance_metrics).transpose()

# Print all the output performance metrics
print(performance_df)

# Save the trained hybrid model and scaler
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('../model', exist_ok=True)

# Save the hybrid model
with open('../model/hybrid_model.pkl', 'wb') as f:
    pickle.dump(hybrid_model, f)

# Save the scaler (from earlier)
with open('../model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")