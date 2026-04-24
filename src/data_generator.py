import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

# Column metadata: type and valid range
COLUMN_META = {
    'age':             {'type': 'int',    'min': 18,  'max': 100},
    'sex':             {'type': 'binary_label', 'labels': {0: 'Female', 1: 'Male'}},
    'chest_pain':      {'type': 'int',    'min': 0,   'max': 3},
    'resting_bp':      {'type': 'int',    'min': 80,  'max': 220},
    'cholesterol':     {'type': 'int',    'min': 100, 'max': 600},
    'fasting_bs':      {'type': 'binary', 'min': 0,   'max': 1},
    'rest_ecg':        {'type': 'int',    'min': 0,   'max': 2},
    'max_hr':          {'type': 'int',    'min': 50,  'max': 220},
    'exercise_angina': {'type': 'binary', 'min': 0,   'max': 1},
    'st_depression':   {'type': 'float',  'min': 0.0, 'max': 8.0},
    'glucose':         {'type': 'int',    'min': 50,  'max': 300},
    'bmi':             {'type': 'float',  'min': 15.0,'max': 60.0},
    'insulin':         {'type': 'int',    'min': 0,   'max': 900},
    'skin_thickness':  {'type': 'int',    'min': 5,   'max': 80},
    'pregnancies':     {'type': 'int',    'min': 0,   'max': 15},
    'dpf':             {'type': 'float',  'min': 0.05,'max': 3.0},
    'heart_disease':   {'type': 'binary', 'min': 0,   'max': 1},
    'diabetes':        {'type': 'binary', 'min': 0,   'max': 1},
    'risk_score':      {'type': 'float',  'min': 0.0, 'max': 1.0},
}

class HealthcareDataGenerator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        self.original_data = None
        self.processed_data = None

    def load_heart_disease(self):
        logger.info("Loading Heart Disease dataset from UCI...")
        try:
            heart = fetch_ucirepo(id=45)
            df = heart.data.features.copy()
            df['target'] = heart.data.targets.values.ravel()
            df['target'] = (df['target'] > 0).astype(int)
            df.dropna(inplace=True)
            logger.info(f"Heart Disease dataset loaded: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"UCI fetch failed: {e}. Using fallback.")
            return self._generate_heart_fallback()

    def load_diabetes(self):
        logger.info("Loading Diabetes dataset...")
        try:
            diabetes = fetch_ucirepo(id=34)
            df = diabetes.data.features.copy()
            df['target'] = diabetes.data.targets.values.ravel()
            df.dropna(inplace=True)
            logger.info(f"Diabetes dataset loaded: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"UCI fetch failed: {e}. Using fallback.")
            return self._generate_diabetes_fallback()

    def load_combined_dataset(self):
        heart_df = self.load_heart_disease()
        diabetes_df = self.load_diabetes()

        np.random.seed(42)
        n = min(len(heart_df), len(diabetes_df), 300)
        heart_sample = heart_df.sample(n=n, random_state=42).reset_index(drop=True)
        diab_sample  = diabetes_df.sample(n=n, random_state=42).reset_index(drop=True)

        def _col(df, name, fallback):
            return df[name] if name in df.columns else fallback

        combined = pd.DataFrame({
            'age':             _col(heart_sample, 'age',      np.random.randint(30, 80, n)).astype(int),
            'sex':             _col(heart_sample, 'sex',      np.random.randint(0, 2, n)).astype(int),
            'chest_pain':      _col(heart_sample, 'cp',       np.random.randint(0, 4, n)).astype(int),
            'resting_bp':      _col(heart_sample, 'trestbps', np.random.randint(90, 180, n)).astype(int),
            'cholesterol':     _col(heart_sample, 'chol',     np.random.randint(150, 350, n)).astype(int),
            'fasting_bs':      _col(heart_sample, 'fbs',      np.random.randint(0, 2, n)).astype(int),
            'rest_ecg':        _col(heart_sample, 'restecg',  np.random.randint(0, 3, n)).astype(int),
            'max_hr':          _col(heart_sample, 'thalach',  np.random.randint(70, 200, n)).astype(int),
            'exercise_angina': _col(heart_sample, 'exang',    np.random.randint(0, 2, n)).astype(int),
            'st_depression':   _col(heart_sample, 'oldpeak',  np.random.uniform(0, 5, n)).round(1),
            'glucose':         _col(diab_sample,  'Glucose',  np.random.randint(70, 200, n)).astype(int),
            'bmi':             _col(diab_sample,  'BMI',      np.random.uniform(18, 45, n)).round(1),
            'insulin':         _col(diab_sample,  'Insulin',  np.random.randint(0, 300, n)).astype(int),
            'skin_thickness':  _col(diab_sample,  'SkinThickness', np.random.randint(10, 50, n)).astype(int),
            'pregnancies':     _col(diab_sample,  'Pregnancies',   np.random.randint(0, 12, n)).astype(int),
            'dpf':             _col(diab_sample,  'DiabetesPedigreeFunction', np.random.uniform(0.1, 2.5, n)).round(3),
            'heart_disease':   _col(heart_sample, 'target',   np.random.randint(0, 2, n)).astype(int),
            'diabetes':        _col(diab_sample,  'target',   np.random.randint(0, 2, n)).astype(int),
        })

        combined['risk_score'] = (
            0.3 * combined['heart_disease'] +
            0.3 * combined['diabetes'] +
            0.2 * (combined['age'] > 55).astype(int) +
            0.1 * (combined['cholesterol'] > 240).astype(int) +
            0.1 * (combined['bmi'] > 30).astype(int)
        ).round(3)

        self.original_data = combined
        return combined

    def preprocess(self, df):
        self.feature_names = list(df.columns)
        processed = df.copy()
        for col in processed.columns:
            processed[col] = pd.to_numeric(processed[col], errors='coerce')
        processed.fillna(processed.median(), inplace=True)
        scaled = self.scaler.fit_transform(processed)
        self.processed_data = pd.DataFrame(scaled, columns=self.feature_names)
        return self.processed_data

    def inverse_transform(self, data_array):
        try:
            return self.scaler.inverse_transform(data_array)
        except Exception:
            return data_array

    def postprocess_synthetic(self, df_raw):
        """
        Takes inverse-transformed DataFrame and snaps every column
        to its correct type and valid clinical range.
        Also converts binary columns to human-readable labels.
        """
        df = df_raw.copy()
        display_df = df_raw.copy()

        for col in df.columns:
            if col not in COLUMN_META:
                continue
            meta = COLUMN_META[col]
            ctype = meta['type']

            if ctype == 'binary':
                # Round to nearest 0 or 1, clip
                df[col]         = df[col].round().clip(0, 1).astype(int)
                display_df[col] = df[col]

            elif ctype == 'binary_label':
                # Round to 0/1 then map to label
                df[col]         = df[col].round().clip(0, 1).astype(int)
                display_df[col] = df[col].map(meta['labels'])

            elif ctype == 'int':
                df[col]         = df[col].round().clip(meta['min'], meta['max']).astype(int)
                display_df[col] = df[col]

            elif ctype == 'float':
                df[col]         = df[col].clip(meta['min'], meta['max']).round(2)
                display_df[col] = df[col]

        return df, display_df   # numeric_df for ML, display_df for UI

    def to_dataframe(self, array):
        return pd.DataFrame(array, columns=self.feature_names)

    # ── fallbacks ──────────────────────────────────────────────────────────
    def _generate_heart_fallback(self):
        np.random.seed(42)
        n = 303
        return pd.DataFrame({
            'age':      np.random.randint(29, 77, n),
            'sex':      np.random.randint(0, 2, n),
            'cp':       np.random.randint(0, 4, n),
            'trestbps': np.random.randint(94, 200, n),
            'chol':     np.random.randint(126, 564, n),
            'fbs':      np.random.randint(0, 2, n),
            'restecg':  np.random.randint(0, 3, n),
            'thalach':  np.random.randint(71, 202, n),
            'exang':    np.random.randint(0, 2, n),
            'oldpeak':  np.random.uniform(0, 6.2, n).round(1),
            'target':   np.random.randint(0, 2, n),
        })

    def _generate_diabetes_fallback(self):
        np.random.seed(99)
        n = 768
        return pd.DataFrame({
            'Pregnancies':               np.random.randint(0, 17, n),
            'Glucose':                   np.random.randint(44, 199, n),
            'BloodPressure':             np.random.randint(24, 122, n),
            'SkinThickness':             np.random.randint(7, 99, n),
            'Insulin':                   np.random.randint(14, 846, n),
            'BMI':                       np.random.uniform(18.2, 67.1, n).round(1),
            'DiabetesPedigreeFunction':  np.random.uniform(0.078, 2.42, n).round(3),
            'Age':                       np.random.randint(21, 81, n),
            'target':                    np.random.randint(0, 2, n),
        })
