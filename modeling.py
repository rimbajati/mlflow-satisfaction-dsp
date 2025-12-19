import os
import warnings
import sys

import matplotlib
matplotlib.use('Agg')

from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
from mlflow.client import MlflowClient
import mlflow.sklearn

from mlflow.models import infer_signature
from sklearn.preprocessing import LabelEncoder
import joblib 

warnings.filterwarnings("ignore")
load_dotenv()

TOP_FEATURES_LIST = [
    'Online boarding',
    'Inflight wifi service',
    'Class',
    'Type of Travel',
    'Inflight entertainment',
    'Seat comfort',
    'Leg room service',
    'On-board service',
    'Flight Distance',
    'Customer Type'
]

def run_rf_model_mlflow(df, top_features):
    
    uri_dagshub = "https://dagshub.com/rimbajati/MLflow-satisfaction.mlflow" 
    mlflow.set_tracking_uri(uri_dagshub)
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

    experiment_name = "airline_satisfaction_AUTO_V6" 
    
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"🔨 Membuat eksperimen BARU: {experiment_name}")
        experiment_id = client.create_experiment(name=experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    else:
        print(f"📂 Menggunakan eksperimen: '{experiment_name}'")
        experiment_id = experiment.experiment_id

    df_processed = df.copy()
    
    target_counts = df_processed['satisfaction'].value_counts()
    valid_classes = target_counts[target_counts > 1].index
    df_processed = df_processed[df_processed['satisfaction'].isin(valid_classes)]
    
    print("⚙️ Encoding Data...")
    for col in df_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
    
    y = df_processed['satisfaction'].astype(int) 
    X_full = df_processed.drop('satisfaction', axis=1)

    try:
        X_top = X_full[top_features]
    except KeyError as e:
        print(f" Error Fitur: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, stratify=y, random_state=42)

    print("🚀 Mulai Training & Upload ke DagsHub...")

    with mlflow.start_run(run_name="rf-auto-signature", experiment_id=experiment.experiment_id) as run:
        

        model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
        model_rf.fit(X_train, y_train)

        y_pred = model_rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f" Accuracy: {test_accuracy:.4f}")
        

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred, zero_division=0))
        
        input_example = X_train.head(5)
        
        prediction_example = model_rf.predict(input_example)
        
        signature = infer_signature(model_input=input_example, model_output=prediction_example)
        
        registered_model_name = "airline_satisfaction_rf" 

        print(" Mengupload Model ke DagsHub...")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model=model_rf,
                artifact_path="model",
                signature=signature,           
                input_example=input_example,   
                registered_model_name=registered_model_name
            )
            print(f" SUKSES! Model terdaftar sebagai: '{registered_model_name}'")
            print(f" Link Eksperimen: {uri_dagshub}/#/experiments/{experiment_id}")
        else:
            mlflow.sklearn.log_model(
            sk_model=model_rf, 
            artifact_path="model", 
            registered_model_name="airline_satisfaction_rf"
        )

        print(f"Run ID: {run.info.run_id}")

    print("\n Menyimpan Backup Lokal...")
    joblib.dump(model_rf, 'model_airline.pkl')
    print(" File 'model_airline.pkl' siap!")

if __name__ == "__main__":
    dataset_path = "data/data_clean_looker.csv" 
    if os.path.exists(dataset_path):
        try:
            df = pd.read_csv(dataset_path)
            print(f" Dataset dimuat.")
            run_rf_model_mlflow(df, TOP_FEATURES_LIST)
        except Exception as e:
            print(f" Error CSV: {e}")
    else:
        print(f" Error: File '{dataset_path}' tidak ditemukan.")