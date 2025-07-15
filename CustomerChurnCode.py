#JS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# 1. Load and preprocess data
df = pd.read_csv(r"C:\Users\Jashmina\OneDrive\Documents\WA_Fn-UseC_-Telco-Customer-Churn.csv")

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].mean()}, inplace=True)
df.drop(['customerID'], axis=1, inplace=True)

categorical = df.select_dtypes(include='object').columns.tolist()
categorical.remove('Churn')
df = pd.get_dummies(df, columns=categorical, drop_first=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

scaler = MinMaxScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# 2. Baseline models (no SMOTE)
lr_base = LogisticRegression(max_iter=1000)
lr_base.fit(X_train, y_train)
y_pred_lr_base = lr_base.predict(X_val)
y_prob_lr_base = lr_base.predict_proba(X_val)[:,1]

rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_rf_base = rf_base.predict(X_val)
y_prob_rf_base = rf_base.predict_proba(X_val)[:,1]

# 3. SMOTE models with GridSearchCV
smote = SMOTE(random_state=42)

lr = LogisticRegression(max_iter=1000)
param_grid_lr = {'model__C': [0.01, 0.1, 1, 10]}
pipe_lr = Pipeline([('smote', smote), ('model', lr)])
grid_lr = GridSearchCV(pipe_lr, param_grid=param_grid_lr, scoring='f1', cv=5)
grid_lr.fit(X_train, y_train)
y_pred_lr_smote = grid_lr.predict(X_val)
y_prob_lr_smote = grid_lr.predict_proba(X_val)[:,1]

rf = RandomForestClassifier(random_state=42)
param_grid_rf = {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20, None]}
pipe_rf = Pipeline([('smote', smote), ('model', rf)])
grid_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, scoring='f1', cv=5)
grid_rf.fit(X_train, y_train)
y_pred_rf_smote = grid_rf.predict(X_val)
y_prob_rf_smote = grid_rf.predict_proba(X_val)[:,1]

# Helper function to plot confusion matrix and save
def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(filename)
    plt.show()

# 4. Print classification reports
print("Logistic Regression Baseline:\n", classification_report(y_val, y_pred_lr_base))
print("Random Forest Baseline:\n", classification_report(y_val, y_pred_rf_base))
print("Logistic Regression SMOTE:\n", classification_report(y_val, y_pred_lr_smote))
print("Random Forest SMOTE:\n", classification_report(y_val, y_pred_rf_smote))

# 5. Plot confusion matrices
plot_confusion_matrix(y_val, y_pred_lr_base, "Confusion Matrix - Logistic Regression Baseline", "confusion_lr_baseline.png")
plot_confusion_matrix(y_val, y_pred_lr_smote, "Confusion Matrix - Logistic Regression SMOTE", "confusion_lr_smote.png")
plot_confusion_matrix(y_val, y_pred_rf_base, "Confusion Matrix - Random Forest Baseline", "confusion_rf_baseline.png")
plot_confusion_matrix(y_val, y_pred_rf_smote, "Confusion Matrix - Random Forest SMOTE", "confusion_rf_smote.png")

# 6. Plot ROC curves comparison helper
def plot_roc_comparison(y_true, prob1, label1, prob2, label2, title, filename):
    fpr1, tpr1, _ = roc_curve(y_true, prob1)
    fpr2, tpr2, _ = roc_curve(y_true, prob2)
    auc1 = auc(fpr1, tpr1)
    auc2 = auc(fpr2, tpr2)

    plt.figure()
    plt.plot(fpr1, tpr1, label=f'{label1} (AUC = {auc1:.3f})')
    plt.plot(fpr2, tpr2, label=f'{label2} (AUC = {auc2:.3f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.show()

# 7. Plot ROC curves
plot_roc_comparison(y_val, y_prob_lr_base, 'Baseline', y_prob_lr_smote, 'SMOTE', 'ROC Curve - Logistic Regression', 'roc_lr.png')
plot_roc_comparison(y_val, y_prob_rf_base, 'Baseline', y_prob_rf_smote, 'SMOTE', 'ROC Curve - Random Forest', 'roc_rf.png')

# 8. Feature importance for Random Forest SMOTE
importances = grid_rf.best_estimator_.named_steps['model'].feature_importances_
features = X_train.columns

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Random Forest Feature Importance (SMOTE)")
plt.tight_layout()
plt.savefig('feature_importance_rf_smote.png')
plt.show()
