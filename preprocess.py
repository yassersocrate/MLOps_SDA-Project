import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Import SMOTE for handling class imbalance


def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)

    # Handling missing values (example: fill NA with mean)
    df.fillna(df.mean(), inplace=True)

    # Optional: Feature engineering or adding/removing features

    # Separating target variable 'default' and features
    X = df.drop(columns=["default", "customer_id"])  # Drop the target and customer_id
    y = df["default"]

    # Feature scaling (standardize numerical features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Return resampled training data and original test data
    return X_train_resampled, X_test, y_train_resampled, y_test
