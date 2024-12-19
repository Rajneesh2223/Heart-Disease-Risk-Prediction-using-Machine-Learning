import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/heart_disease_data.csv')

# Handle missing values
df = df.dropna()  # or apply imputation

# Feature scaling
scaler = StandardScaler()
X = df.drop(columns=['target'])  # assuming 'target' is the column for heart disease outcome
y = df['target']
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
