import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

# Load the training data
df = pd.read_csv('/kaggle/input/predict-the-success-of-bank-telemarketing/train.csv')

# Preprocessing function (only using top contributing features)
def preprocess_data(data, is_training=True):
    # Check if the dataframe is loaded properly
    print("Data loaded, shape:", data.shape)

    # Convert 'last contact date' to datetime and create new date-related features
    data['last contact date'] = pd.to_datetime(data['last contact date'], errors='coerce')
    data['contact_month'] = data['last contact date'].dt.month
    data['contact_day'] = data['last contact date'].dt.day
    data['contact_weekday'] = data['last contact date'].dt.weekday

    # Select only important features
    selected_features = ['duration', 'balance', 'age', 'pdays', 'previous', 
                         'contact_month', 'contact_day', 'contact_weekday']
    
    if 'target' in data.columns and is_training:
        selected_features.append('target')

    # Filter the data for the selected features
    data = data[selected_features]
    print("Data after selecting important features, shape:", data.shape)

    # Handle missing values
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    # Impute missing numeric values
    numeric_imputer = SimpleImputer(strategy='median')
    data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])
    
    return data

# Feature engineering and preprocessing for training data
df = preprocess_data(df, is_training=True)

# Prepare features and target
X = df.drop('target', axis=1)
y = df['target']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data using stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate Precision-Recall AUC
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')
print(f"\nCross-validation ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Print feature importances
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

# Load the test data
test_df = pd.read_csv('/kaggle/input/predict-the-success-of-bank-telemarketing/test.csv')

# Preprocess the test data (with selected features only)
test_df = preprocess_data(test_df, is_training=False)

# Make predictions on the test data
test_predictions = rf_model.predict(test_df)

# Convert numeric predictions back to 'yes' or 'no'
test_predictions = le.inverse_transform(test_predictions)

# Create a DataFrame with the predictions
submission = pd.DataFrame({
    'id': range(len(test_predictions)),  # Generate IDs if not present
    'target': test_predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' has been created.")