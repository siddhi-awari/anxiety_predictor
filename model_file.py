import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

# Generate dummy training data
X_train = np.random.randint(0, 5, size=(100, 10))  # Features
y_train = np.random.choice(["Mild", "Moderate", "Severe"], size=100)  # Labels

# Train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the model
with open("model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)
