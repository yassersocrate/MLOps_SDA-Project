import mlflow
import mlflow.sklearn
from preprocess import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import argparse
import joblib
import os
import json


# Function to save hyperparameters to a text file
def save_hyperparameters_to_txt(file_name, best_params):
    with open(file_name, "w") as file:
        for param, value in best_params.items():
            file.write(f"{param}: {value}\n")


# Function to save metrics to a JSON file
def save_metrics_to_json(file_name, metrics):
    with open(file_name, "w") as file:
        json.dump(metrics, file)


# Function to train Random Forest model with or without hyperparameter tuning
def train_random_forest(X_train, y_train, tune_hyperparameters):
    if tune_hyperparameters:
        # Define parameter grid for Random Forest
        param_grid_rf = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        rf_model = RandomForestClassifier(random_state=42)
        grid_search_rf = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid_rf,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring="roc_auc",
        )
        grid_search_rf.fit(X_train, y_train)

        print(f"Best Random Forest Hyperparameters: {grid_search_rf.best_params_}")

        # Save best hyperparameters to a text file
        save_hyperparameters_to_txt(
            "rf_best_hyperparams.txt", grid_search_rf.best_params_
        )

        return grid_search_rf.best_estimator_
    else:
        with open("rf_best_hyperparams.txt", "r") as file:
            best_params = dict(line.strip().split(": ") for line in file)

        # Convert string values to the appropriate types (e.g., integers, floats)
        best_params = {
            k: (int(v) if v.isdigit() else float(v) if "." in v else v)
            for k, v in best_params.items()
        }

        # Use the parameters to initialize the model
        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(X_train, y_train)
        return rf_model


# Function to train XGBoost model with or without hyperparameter tuning
def train_xgboost(X_train, y_train, tune_hyperparameters):
    if tune_hyperparameters:
        # Define parameter grid for XGBoost
        param_grid_xgb = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

        xgb_model = XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
        grid_search_xgb = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid_xgb,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring="roc_auc",
        )
        grid_search_xgb.fit(X_train, y_train)

        print(f"Best XGBoost Hyperparameters: {grid_search_xgb.best_params_}")

        # Save best hyperparameters to a text file
        save_hyperparameters_to_txt(
            "xgb_best_hyperparams.txt", grid_search_xgb.best_params_
        )

        return grid_search_xgb.best_estimator_
    else:
        with open("xgb_best_hyperparams.txt", "r") as file:
            best_params = dict(line.strip().split(": ") for line in file)

        # Convert string values to the appropriate types (e.g., integers, floats)
        best_params = {
            k: (int(v) if v.isdigit() else float(v) if "." in v else v)
            for k, v in best_params.items()
        }

        # Use the parameters to initialize the XGBoost model
        xgb_model = XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            **best_params,
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model


# Function to evaluate model and display detailed metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # For ROC AUC calculation

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Classification report (as a string, not needed to store in JSON or MLflow)
    class_report = classification_report(y_test, y_pred)

    # Print metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Confusion Matrix:\n {conf_matrix}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Classification Report:\n {class_report}")

    # Return metrics as a dictionary
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": conf_matrix.tolist(),  # Convert to list for JSON compatibility
    }
    return metrics


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        required=True,
        help="Choose whether to train or evaluate the model.",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "xgboost"],
        required=False,
        help="Choose the model to train (random_forest or xgboost)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Whether to tune hyperparameters during training",
    )
    parser.add_argument(
        "--save_model", action="store_true", help="Whether to save the trained model"
    )
    parser.add_argument(
        "--load_model", type=str, help="Path to the saved model for evaluation"
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default="metrics.json",
        help="Path to save metrics as JSON",
    )
    args = parser.parse_args()

    if args.mode == "train":
        # Load and preprocess data
        X_train, X_test, y_train, y_test = load_and_preprocess_data("Loan_Data.csv")

        # Start MLflow experiment
        mlflow.set_experiment("Loan Default Prediction with Hyperparameter Tuning")

        if args.model == "random_forest":
            with mlflow.start_run(
                run_name=(
                    "Random Forest with Tuning"
                    if args.tune
                    else "Random Forest without Tuning"
                )
            ) as run_rf:
                print(
                    f"Training Random Forest {'with' if args.tune else 'without'} Hyperparameter Tuning..."
                )
                rf_model = train_random_forest(X_train, y_train, args.tune)

                # Evaluate model
                metrics = evaluate_model(rf_model, X_test, y_test)

                # Log metrics to MLflow
                mlflow.log_param(
                    "model_type",
                    f"Random Forest {'with' if args.tune else 'without'} Tuning",
                )
                for key, value in metrics.items():
                    if (
                        key != "confusion_matrix"
                    ):  # Avoid logging confusion matrix to MLflow
                        mlflow.log_metric(key, value)

                mlflow.sklearn.log_model(rf_model, "random_forest_model")

                # Optionally save the model for future evaluation
                if args.save_model:
                    joblib.dump(rf_model, "models/random_forest_model.pkl")
                    print("Random Forest model saved to random_forest_model.pkl")

                # Save metrics to JSON
                save_metrics_to_json(args.save_metrics, metrics)

        elif args.model == "xgboost":
            with mlflow.start_run(
                run_name=(
                    "XGBoost with Tuning" if args.tune else "XGBoost without Tuning"
                )
            ) as run_xgb:
                print(
                    f"Training XGBoost {'with' if args.tune else 'without'} Hyperparameter Tuning..."
                )
                xgb_model = train_xgboost(X_train, y_train, args.tune)

                # Evaluate model
                metrics = evaluate_model(xgb_model, X_test, y_test)

                # Log metrics to MLflow
                mlflow.log_param(
                    "model_type", f"XGBoost {'with' if args.tune else 'without'} Tuning"
                )
                for key, value in metrics.items():
                    if key != "confusion_matrix":
                        mlflow.log_metric(key, value)

                mlflow.sklearn.log_model(xgb_model, "xgboost_model")

                # Optionally save the model for future evaluation
                if args.save_model:
                    joblib.dump(xgb_model, "models/xgb_model.pkl")
                    print("XGBoost model saved to xgb_model.pkl")

                # Save metrics to JSON
                save_metrics_to_json(args.save_metrics, metrics)

    elif args.mode == "eval":
        if not args.load_model:
            print("Please provide a model file path to load using --load_model")
        else:
            # Load model from the specified path
            model = joblib.load(args.load_model)

            # Load and preprocess data
            X_train, X_test, y_train, y_test = load_and_preprocess_data("Loan_Data.csv")

            # Evaluate the loaded model
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics to MLflow during evaluation
            with mlflow.start_run(run_name="Xgboost Eval"): 
                for key, value in metrics.items():
                    if key != "confusion_matrix":
                        mlflow.log_metric(key, value)

            # Save metrics to JSON
            save_metrics_to_json(args.save_metrics, metrics)
