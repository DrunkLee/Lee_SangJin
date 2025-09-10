import pandas as pd
import numpy as np
import joblib

class HybridPreprocessor:
    def __init__(self, n_bins=10):
        self.eps = 1e-8
        self.n_bins = n_bins
        self.encoding_maps = {}
        self.interaction_features = []
        self.final_features = []
        self.numeric_cols = ['age', 'tenure', 'frequent', 'payment_interval', 'contract_length', 'after_interaction']
        self.categorical_cols = ['gender', 'subscription_type']

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        df_interact = df[self.numeric_cols].copy()
        df_dummies = pd.get_dummies(df[self.categorical_cols])
        df_interact["age_F"] = df_interact["age"] * df_dummies.get("gender_F", 0)
        df_interact["age_M"] = df_interact["age"] * df_dummies.get("gender_M", 0)
        df_interact["age_F_frequent"] = df_interact["age_F"] * df_interact["frequent"]
        df_interact["age_M_frequent"] = df_interact["age_M"] * df_interact["frequent"]
        df_interact["tenure_vip"] = df_interact["tenure"] * df_dummies.get("subscription_type_vip", 0)
        df_interact["tenure_plus"] = df_interact["tenure"] * df_dummies.get("subscription_type_plus", 0)
        df_interact["tenure_member"] = df_interact["tenure"] * df_dummies.get("subscription_type_member", 0)

        df_interact["after_contract"] = df_interact["after_interaction"] * df_interact["contract_length"]
        df_interact["age_contract"] = df_interact["age"] * df_interact["contract_length"]
        df_interact["activity_per_tenure"] = df_interact["frequent"] / (df_interact["tenure"] + self.eps)
        df_interact["contract_per_freq"] = df_interact["contract_length"] / (df_interact["frequent"] + self.eps)
        df_interact["payment_per_contract"] = df_interact['payment_interval'] / (df_interact['contract_length'] + self.eps)
        df_interact['age_x_payment'] = df_interact['age'] * df_interact['payment_interval']
        df_interact['inactivity_ratio'] = df_interact['after_interaction'] / (df_interact['tenure'] + self.eps)
        df_interact['tenure_x_frequent'] = df_interact['tenure'] * df_interact['frequent']

        self.interaction_features = [col for col in df_interact.columns if col not in self.numeric_cols]
        
        return df_interact[self.interaction_features]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """학습 데이터(X, y)로 인코딩 규칙을 학습합니다."""
        print("[INFO] 데이터 사전 학습 시작.")
        df_interacted = self._create_interaction_features(X)
        temp_binned_df = pd.DataFrame()
        for col in self.interaction_features:
            binned_col_name = f"{col}_binned"
            try:
                temp_binned_df[binned_col_name], bins = pd.qcut(df_interacted[col], q=self.n_bins, retbins=True, duplicates='drop', labels=False)
            except ValueError:
                bins = np.linspace(df_interacted[col].min(), df_interacted[col].max(), self.n_bins + 1)
            temp_binned_df.index = y.index
            target_mean_map = temp_binned_df.groupby(binned_col_name)[y.name].mean()
            self.encoding_maps[col] = {'bins': bins, 'map': target_mean_map}
        self.encoding_maps['global_mean'] = y.mean()
        
        self.final_features = list(X.columns) + self.interaction_features + [f"{col}_TE" for col in self.interaction_features]
        print("[INFO] 데이터 사전 학습 완료.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """학습된 규칙으로 데이터를 변환합니다. (기존 피처 + 상호작용 피처 + 확률형 피처)"""
        if not self.encoding_maps:
            raise RuntimeError("[ERROR] 데이터를 사전학습하지 않아 실행이 불가합니다.")
        df_interacted = self._create_interaction_features(X)
        df_target_encoded = pd.DataFrame(index=X.index)
        for col in self.interaction_features:
            encoded_col_name = f"{col}_TE"
            rules = self.encoding_maps[col]
            
            binned_series = pd.cut(df_interacted[col], bins=rules['bins'], labels=False, include_lowest=True)
            df_target_encoded[encoded_col_name] = binned_series.map(rules['map'])
            df_target_encoded[encoded_col_name].fillna(self.encoding_maps['global_mean'], inplace=True)
        X_final = pd.get_dummies(X, columns=self.categorical_cols)
        X_final = pd.concat([X_final, df_interacted, df_target_encoded], axis=1)
        return X_final.reindex(columns=self.final_features, fill_value=0)