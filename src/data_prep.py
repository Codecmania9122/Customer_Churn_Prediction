# src/data_prep.py
import pandas as pd
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os


class DataPreprocessor:
    def __init__(self, config_path: str):
        # Load YAML config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.categorical_cols = self.config["preprocessing"]["categorical_cols"]
        self.numeric_cols = self.config["preprocessing"]["numeric_cols"]
        self.target_col = self.config["preprocessing"]["target_col"]

        # Scalers & encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Encode categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Scale numeric columns
        df[self.numeric_cols] = self.scaler.fit_transform(df[self.numeric_cols])

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Apply fitted encoders
        for col in self.categorical_cols:
            le = self.label_encoders.get(col)
            if le:
                df[col] = le.transform(df[col].astype(str))

        # Apply scaler
        df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])

        return df

    def save(self, output_dir="models/preprocessors"):
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.label_encoders, os.path.join(output_dir, "label_encoders.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))

    def load(self, input_dir="models/preprocessors"):
        self.label_encoders = joblib.load(os.path.join(input_dir, "label_encoders.pkl"))
        self.scaler = joblib.load(os.path.join(input_dir, "scaler.pkl"))
