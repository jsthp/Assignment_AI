import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import numpy as np # Importing numpy for the confusion matrix labeling

# 1. Load and preprocess data
df = pd.read_csv(r"C:\Users\Jashmina\Downloads\Telco-Customer-Churn.csv")

# Defines the 9 key features as per your updated methodology, including 'Churn'
# These are the ORIGINAL column names before any encoding.
selected_original_features = [
    'gender', 'SeniorCitizen', 'Partner', 'tenure', 'MonthlyCharges',
    'Contract', 'InternetService', 'PaymentMethod', 'Churn'
]

# This shows the raw data as loaded, before any major changes or feature selection.
print("--- Initial Data (Head) ---")
print(df[selected_original_features].head()) # Show head of only the *selected* features for clarity
print("\n--- Initial Data (Info for Selected Features) ---")
print(df[selected_original_features].info()) # Info for only the *selected* features
print("\n--- Initial Data (Missing Values for Selected Features) ---")
print(df[selected_original_features].isnull().sum()) # Missing values for *selected* features


# Selects only these 9 features from the DataFrame and create a copy to avoid warnings
df = df[selected_original_features].copy()

# Drops 'customerID' if it exists (it won't if only selected 9 features above, but as a safeguard)
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# Handles 'TotalCharges' as per updated methodology: it's dropped because it's not in the 9 selected features
if 'TotalCharges' in df.columns:
    df.drop('TotalCharges', axis=1, inplace=True)

# Converts 'Churn' target variable to numerical (Yes=1, No=0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Identifies categorical columns for one-hot encoding *after* feature selection, excluding 'Churn'
categorical_features_for_encoding = df.select_dtypes(include='object').columns.tolist()

# Performs one-hot encoding on the identified categorical features
df = pd.get_dummies(df, columns=categorical_features_for_encoding, drop_first=True)

# This shows how the categorical features have been converted into numerical (0/1) columns.
print("\n--- Data After One-Hot Encoding (Head) ---")
print(df.head())
print("\n--- Data After One-Hot Encoding (Columns) ---")
print(df.columns.tolist()) # Shows all the new and old columns after encoding

# Normalization: Numerical features (tenure and MonthlyCharges) as per updated methodology
scaler = MinMaxScaler()
# Ensures only 'tenure' and 'MonthlyCharges' are scaled, as per your reduced feature set
df[['tenure', 'MonthlyCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges']])

# This shows the numerical features scaled to a 0-1 range.
print("\n--- Data After Normalization (Head, relevant columns) ---")
print(df[['tenure', 'MonthlyCharges']].head())
print("\n--- Data After Normalization (Descriptive Stats, relevant columns) ---")
print(df[['tenure', 'MonthlyCharges']].describe()) # Min should be 0, Max should be 1

# Prepares features (X) and target (y) for model training
X = df.drop('Churn', axis=1)
y = df['Churn']

# Splits data into training (70%), validation (15%), and test (15%) sets using stratified sampling
X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# --- NEWLY ADDED: Visualize Class Distribution BEFORE and AFTER SMOTE ---
# This part is for generating the plots for your report
print("\n--- Generating Class Distribution Plots ---")
# Visualize Class Distribution BEFORE SMOTE
plt.figure(figsize=(6, 5))
sns.countplot(x=y_train)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Churn Status (0: No Churn, 1: Churn)')
plt.ylabel('Number of Samples')
plt.savefig('class_distribution_before_smote.png') # Save the plot
plt.show()

# Apply SMOTE for Visualization Purposes (temporarily outside pipeline to get resampled y)
smote_visual = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote_visual.fit_resample(X_train, y_train)

# Visualize Class Distribution AFTER SMOTE
plt.figure(figsize=(6, 5))
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Churn Status (0: No Churn, 1: Churn)')
plt.ylabel('Number of Samples')
plt.savefig('class_distribution_after_smote.png') # Save the plot
plt.show()

# Optional: Print counts to verify
print("\nClass distribution before SMOTE:")
print(y_train.value_counts())
print("\nClass distribution after SMOTE:")
print(y_train_resampled.value_counts())
# --- END NEWLY ADDED CODE ---


# 2. Baseline Logistic Regression model (no SMOTE)
lr_base = LogisticRegression(max_iter=1000, random_state=42) # Added random_state for reproducibility
lr_base.fit(X_train, y_train)
y_pred_lr_base = lr_base.predict(X_val)
y_prob_lr_base = lr_base.predict_proba(X_val)[:,1]

# 3. Logistic Regression with SMOTE and GridSearchCV
smote = SMOTE(random_state=42) # Initialize SMOTE

lr_model = LogisticRegression(max_iter=1000, random_state=42) # Added random_state
param_grid_lr = {'model__C': [0.01, 0.1, 1, 10]} # Hyperparameter C for Logistic Regression

# Creates a pipeline that first applies SMOTE and then trains the Logistic Regression model
pipe_lr = Pipeline([('smote', smote), ('model', lr_model)])

# Uses GridSearchCV to find the best hyperparameters for the pipeline
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, scoring='f1', cv=5)
grid_lr.fit(X_train, y_train) # Train the GridSearchCV pipeline on the training data

# Makes predictions on the validation set using the best model from GridSearchCV
y_pred_lr_smote = grid_lr.predict(X_val)
y_prob_lr_smote = grid_lr.predict_proba(X_val)[:,1]

# Helper function to plot confusion matrix and save (Modified to put labels inside boxes)
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    
    # Creates the annotation array with counts and labels
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    
    # Combines counts and labels for annotation
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_names)]
    labels = np.asarray(labels).reshape(2,2) # Reshape back to 2x2 matrix
    
    plt.figure(figsize=(7,6)) # Increased figure size for better readability
    sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False, # Use 'labels' for annot and '' for fmt
                xticklabels=['Predicted No Churn (0)', 'Predicted Churn (1)'],
                yticklabels=['Actual No Churn (0)', 'Actual Churn (1)'],
                annot_kws={"size": 14}) # Font size for annotations
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# 4. Prints classification reports for both baseline and SMOTE-enhanced LR
print("\n--- Classification Report - Logistic Regression Baseline ---")
print(classification_report(y_val, y_pred_lr_base))
print("\n--- Classification Report - Logistic Regression with SMOTE ---")
print(classification_report(y_val, y_pred_lr_smote))

# 5. Plots confusion matrices for both baseline and SMOTE-enhanced LR
plot_confusion_matrix(y_val, y_pred_lr_base, "Confusion Matrix - Logistic Regression Baseline", "confusion_lr_baseline.png")
plot_confusion_matrix(y_val, y_pred_lr_smote, "Confusion Matrix - Logistic Regression with SMOTE", "confusion_lr_smote.png")

# Helper function to plot ROC curves comparison
def plot_roc_comparison(y_true, prob1, label1, prob2, label2, title, filename):
    fpr1, tpr1, _ = roc_curve(y_true, prob1)
    auc1 = auc(fpr1, tpr1)
    
    fpr2, tpr2, _ = roc_curve(y_true, prob2)
    auc2 = auc(fpr2, tpr2)

    plt.figure()
    plt.plot(fpr1, tpr1, label=f'{label1} (AUC = {auc1:.3f})')
    plt.plot(fpr2, tpr2, label=f'{label2} (AUC = {auc2:.3f})')
    plt.plot([0,1], [0,1], 'k--') # Diagonal dashed line for reference (random classifier)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# 7. Plots ROC curves for Logistic Regression Baseline vs. SMOTE
plot_roc_comparison(y_val, y_prob_lr_base, 'Baseline', y_prob_lr_smote, 'SMOTE', 'ROC Curve - Logistic Regression Performance', 'roc_lr.png')

# 8. Plots Feature Coefficients for Logistic Regression SMOTE Model (for interpretability)
print("\n--- Logistic Regression SMOTE Model Feature Coefficients ---")
# Access the Logistic Regression model from the best estimator of the GridSearchCV pipeline
best_lr_model = grid_lr.best_estimator_.named_steps['model']
# Access feature names from the processed X_train data
feature_names = X_train.columns

# Creates a DataFrame for coefficients
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': best_lr_model.coef_[0]
}).sort_values(by='Coefficient', ascending=False) # Sort to easily see most impactful features

print(coefficients_df)

# Plotting coefficients for visual representation in your report
plt.figure(figsize=(10, 7))
sns.barplot(x='Coefficient', y='Feature', data=coefficients_df)
plt.title('Logistic Regression Feature Coefficients (SMOTE)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('lr_coefficients.png') # Save the plot
plt.show()

# Final evaluation on the unseen test set using the best SMOTE model
print("\n--- Final Evaluation on Test Set (Logistic Regression with SMOTE) ---")
y_pred_test = grid_lr.predict(X_test)
y_prob_test = grid_lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_test))
plot_confusion_matrix(y_test, y_pred_test, "Confusion Matrix - LR with SMOTE (Test Set)", "confusion_lr_smote_test.png")
plot_roc_comparison(y_test, lr_base.predict_proba(X_test)[:,1], 'Baseline (Test)', y_prob_test, 'SMOTE (Test)', 'ROC Curve - LR Test Set Performance', 'roc_lr_test.png')