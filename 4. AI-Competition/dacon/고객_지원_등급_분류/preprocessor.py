import pandas as pd
import joblib

class DataPreprocessor:
    def __init__(self):
        self.eps = 1e-8
        
        self.age_bins = [0, 19, 20, 39, 49, 59, 69]
        self.age_labels = ['10s', '20s', '30s', '40s', '50s', '60s']
        
        self.tenure_bins = [0, 5, 25, 60]
        self.tenure_labels = ["short", "medium", "long"]
        
        self.long_inactive_threshold = None
        self.dummy_columns = {}
        
    def make_pickle(self, df: pd.DataFrame, pickle_path: str):
        joblib.dump(df, pickle_path)
    
    def load_pickle(self, pickle_path: str) -> pd.DataFrame:
        return joblib.load(pickle_path)
    
    def fit_encoder(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["age_group"] = pd.cut(df["age"], bins=self.age_bins, labels=self.age_labels, right=False)
        df["tenure_group"] = pd.cut(df["tenure"], bins=self.tenure_bins, labels=self.tenure_labels, right=True)
        
        categorical_cols = ["age_group", "tenure_group", "gender", "subscription_type"]
        
        for col in categorical_cols:
            self.dummy_columns[col] = df[col].unique()
        return self
    
    def transform(self, df: pd.DataFrame, is_train:bool = True) -> pd.DataFrame:
        """
        age: 나이
        gender: 성별
        tenure: 가입 기간
        frequent: 사용 빈도
        payment_interval: 결제 주기
        subscription_type: 구독 유형
        contract_length: 계약 기간
        after_interaction: 마지막 활동 후 경과 시간
        """
        df = df.copy()
        # one-hot
        df['age_group'] = pd.cut(df['age'], bins=self.age_bins, labels=self.age_labels, right=False)
        df["tenure_group"] = pd.cut(df["tenure"], bins=self.tenure_bins, labels=self.tenure_labels, right=True)
        
        categorical_cols = ['age_group', 'tenure_group', 'gender', 'subscription_type']
        for col in categorical_cols:
            if is_train:
                df = pd.get_dummies(df, columns=[col], prefix=col.replace("_group",""),dtype=int)
            else:
                for category in self.dummy_columns[col]:
                    new_col_name = f"{col.replace("_group", "")}_{category}"
                    df[new_col_name] = (df[col] == category).astype(int)
                df = df.drop(columns=[col])
        
        """기본 피쳐"""
        # 나이 x 성별 x 사용 빈도
        df["age_F"] = df["age"] * df["gender_F"]
        df["age_M"] = df["age"] * df["gender_M"]
        df["age_F_frequent"] = df["age_F"] * df["frequent"]
        df["age_M_frequent"] = df["age_M"] * df["frequent"]
        
        # 가입 기간 x 구독 유형
        df["tenure_vip"] = df["tenure"] * df["subscription_type_vip"]
        df["tenure_plus"] = df["tenure"] * df["subscription_type_plus"]
        df["tenure_member"] = df["tenure"] * df["subscription_type_member"]
        
        df["after_contract"] = df["after_interaction"] * df["contract_length"]
        df["age_contract"] = df["age"] * df["contract_length"]
        df["activity_per_tenure"] = df["frequent"] / (df["tenure"] + self.eps)
        df["contract_per_freq"] = df["contract_length"] / (df["frequent"] + self.eps)
        
        df["payment_per_contract"] = df['payment_interval'] / (df['contract_length'] + self.eps)
        df['age_x_payment'] = df['age'] * df['payment_interval']
        df['inactivity_ratio'] = df['after_interaction'] / (df['tenure'] + self.eps)
        df['tenure_x_frequent'] = df['tenure'] * df['frequent']
        
        df = df.drop(columns=['age', 'tenure'])
        print(df.head(3))
        return df
    