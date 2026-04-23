import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class HealthcareDataGenerator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.original_data = None
        self.processed_data = None
        self.categorical_cols = []
        self.numerical_cols = []
        self.target_col = None

    def load_heart_disease(self):
        """Load UCI Heart Disease dataset"""
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
            logger.warning(f"UCI fetch failed: {e}. Using synthetic fallback.")
            return self._generate_heart_fallback()

    def load_diabetes(self):
        """Load Pima Indians Diabetes dataset"""
        logger.info("Loading Diabetes dataset...")
        try:
            diabetes = fetch_ucirepo(id=34)
            df = diabetes.data.features.copy()
            df['target'] = diabetes.data.targets.values.ravel()
            df.dropna(inplace=True)
            logger.info(f"Diabetes dataset loaded: {df.shape}")
            return df
        except Exception as e:
            logger.warning(f"UCI fetch failed: {e}. Using synthetic fallback.")
            return self._generate_diabetes_fallback()

    def load_combined_dataset(self):
        """Combine Heart Disease + Diabetes features for richer patient records"""
        heart_df = self.load_heart_disease()
        diabetes_df = self.load_diabetes()

        np.random.seed(42)
        n = min(len(heart_df), len(diabetes_df), 300)
        heart_sample = heart_df.sample(n=n, random_state=42).reset_index(drop=True)
        diab_sample = diabetes_df.sample(n=n, random_state=42).reset_index(drop=True)

        # Merge features from both datasets
        combined = pd.DataFrame({
            'age': heart_sample.get('age', np.random.randint(30, 80, n)),
            'sex': heart_sample.get('sex', np.random.randint(0, 2, n)),
            'chest_pain': heart_sample.get('cp', np.random.randint(0, 4, n)),
            'resting_bp': heart_sample.get('trestbps', np.random.randint(90, 180, n)),
            'cholesterol': heart_sample.get('chol', np.random.randint(150, 350, n)),
            'fasting_bs': heart_sample.get('fbs', np.random.randint(0, 2, n)),
            'rest_ecg': heart_sample.get('restecg', np.random.randint(0, 3, n)),
            'max_hr': heart_sample.get('thalach', np.random.randint(70, 200, n)),
            'exercise_angina': heart_sample.get('exang', np.random.randint(0, 2, n)),
            'st_depression': heart_sample.get('oldpeak', np.random.uniform(0, 5, n)),
            'glucose': diab_sample.get('Glucose', np.random.randint(70, 200, n)),
            'bmi': diab_sample.get('BMI', np.random.uniform(18, 45, n)),
            'insulin': diab_sample.get('Insulin', np.random.randint(0, 300, n)),
            'skin_thickness': diab_sample.get('SkinThickness', np.random.randint(10, 50, n)),
            'pregnancies': diab_sample.get('Pregnancies', np.random.randint(0, 15, n)),
            'dpf': diab_sample.get('DiabetesPedigreeFunction', np.random.uniform(0.1, 2.5, n)),
            'heart_disease': heart_sample.get('target', np.random.randint(0, 2, n)),
            'diabetes': diab_sample.get('target', np.random.randint(0, 2, n)),
        })

        # Derived risk score
        combined['risk_score'] = (
            0.3 * combined['heart_disease'] +
            0.3 * combined['diabetes'] +
            0.2 * (combined['age'] > 55).astype(int) +
            0.1 * (combined['cholesterol'] > 240).astype(int) +
            0.1 * (combined['bmi'] > 30).astype(int)
        )

        self.original_data = combined
        self.target_col = 'heart_disease'
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
        """Convert scaled data back to original space"""
        try:
            return self.scaler.inverse_transform(data_array)
        except Exception:
            return data_array

    def to_dataframe(self, array):
        return pd.DataFrame(array, columns=self.feature_names)

    def _generate_heart_fallback(self):
        np.random.seed(42)
        n = 303
        return pd.DataFrame({
            'age': np.random.randint(29, 77, n),
            'sex': np.random.randint(0, 2, n),
            'cp': np.random.randint(0, 4, n),
            'trestbps': np.random.randint(94, 200, n),
            'chol': np.random.randint(126, 564, n),
            'fbs': np.random.randint(0, 2, n),
            'restecg': np.random.randint(0, 3, n),
            'thalach': np.random.randint(71, 202, n),
            'exang': np.random.randint(0, 2, n),
            'oldpeak': np.random.uniform(0, 6.2, n),
            'target': np.random.randint(0, 2, n)
        })

    def _generate_diabetes_fallback(self):
        np.random.seed(99)
        n = 768
        return pd.DataFrame({
            'Pregnancies': np.random.randint(0, 17, n),
            'Glucose': np.random.randint(44, 199, n),
            'BloodPressure': np.random.randint(24, 122, n),
            'SkinThickness': np.random.randint(7, 99, n),
            'Insulin': np.random.randint(14, 846, n),
            'BMI': np.random.uniform(18.2, 67.1, n),
            'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n),
            'Age': np.random.randint(21, 81, n),
            'target': np.random.randint(0, 2, n)
        })
