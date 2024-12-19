from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # To save the model

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, 'models/heart_disease_model.pkl')
