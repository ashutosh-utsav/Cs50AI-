# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Preprocessing function
def preprocess_data(data, is_training=True):
    # Ensure we're working with a copy to avoid SettingWithCopyWarning
    data = data.copy()
    
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

    # Separate numeric and categorical columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Handle missing values for numeric columns with median strategy
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy='median')
        data[numeric_columns] = numeric_imputer.fit_transform(data[numeric_columns])

    # Handle missing values for categorical columns with most frequent strategy
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy='most_frequent')
        data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])
    
    return data

# Load and preprocess training data
train_file_path = '/kaggle/input/predict-the-success-of-bank-telemarketing/train.csv'
train_data = pd.read_csv(train_file_path)
train_data = preprocess_data(train_data, is_training=True)

# Separate features and target, and encode the target variable
X = train_data.drop('target', axis=1)
y = train_data['target']
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split and apply SMOTE for class imbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Model 1: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", rf_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Model 2: XGBoost Classifier
xgb_model = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print("XGBoost Accuracy:", xgb_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

# Model 3: Stacking Classifier
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
]
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(max_iter=1000, random_state=42), n_jobs=-1)
stacking_model.fit(X_train_resampled, y_train_resampled)
y_pred_stack = stacking_model.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_pred_stack)
print("Stacking Classifier Accuracy:", stacking_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_stack))

# Model Comparison
model_accuracies = {
    'Random Forest': rf_accuracy,
    'XGBoost': xgb_accuracy,
    'Stacking': stacking_accuracy
}
for model, accuracy in model_accuracies.items():
    print(f"{model} Accuracy: {accuracy:.4f}")

# Choose the best model and prepare submission
best_model_name = max(model_accuracies, key=model_accuracies.get)
print(f"Best Model: {best_model_name} with accuracy {model_accuracies[best_model_name]:.4f}")
best_model = stacking_model if best_model_name == 'Stacking' else (rf_model if best_model_name == 'Random Forest' else xgb_model)

# Load and preprocess test data
test_file_path = '/kaggle/input/predict-the-success-of-bank-telemarketing/test.csv'
test_data = pd.read_csv(test_file_path)
test_data = preprocess_data(test_data, is_training=False)

# Make predictions on the test data and save submission file
test_predictions = best_model.predict(test_data)
test_predictions = le.inverse_transform(test_predictions)
submission = pd.DataFrame({
    'id': range(len(test_predictions)),
    'target': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' has been created.")
