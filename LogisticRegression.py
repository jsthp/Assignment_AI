import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample Data: Hours Studied (X) and Exam Result (y)
# y=0 for Fail, y=1 for Pass
X = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).reshape(-1, 1) # Hours studied
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1]) # Exam result (changed y[3] from 0 to 1)

# Create and train the Logistic Regression model
model_exam = LogisticRegression()
model_exam.fit(X, y)

# Predict probability for a new student who studied 4 hours
new_student_hours = np.array([[4]])
predicted_prob = model_exam.predict_proba(new_student_hours)[0, 1] # Probability of passing
predicted_class = model_exam.predict(new_student_hours)[0] # Predicted class (0 or 1)

print(f"Hours Studied: {new_student_hours[0,0]} hours")
print(f"Predicted Probability of Passing: {predicted_prob:.2f}")
print(f"Predicted Outcome: {'Pass' if predicted_class == 1 else 'Fail'}")


