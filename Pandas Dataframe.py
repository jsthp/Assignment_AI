#JS
import pandas as pd

data = {
    'Trial ID': [1, 2, 3],
    'Model': ['RandomForest', 'RandomForest', 'RandomForest'],
    'n_estimators': [100, 200, 150],
    'max_depth': [10, 5, None],
    'Accuracy': [0.88, 0.85, 0.89],
    'F1-Score': [0.87, 0.84, 0.88],
    'Training Time (s)': [15, 8, 12],
    'Notes': ['Default', 'Reduced depth', 'Optimized']
}
df = pd.DataFrame(data)
print(df)








