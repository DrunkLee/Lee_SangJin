
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.model_selection import StratifiedKFold
from metric import MacroF1Score

import pandas as pd
import torch

from typing import Optional, Dict, Any, List, Tuple

class ModelHandler:
    def __init__(self, 
                 df:           pd.DataFrame, 
                 feature_cols: List[str]                = None, 
                 target_col:   str                      = None,
                 base_model:   str                      = "lgbm",
                 base_parmas:  Optional[Dict[str, Any]] = None,
                 n_classes:    Optional[int]            = None,
                 seed:         int                      = 42,
                 n_splits:     int                      = 5,
                 verbose:      int                      = 1):
        
        if df is None:
            raise ValueError(f"[ERROR][ModelHandler] DataFrame이 입력되지 않았습니다.")
        elif df.empty:
            raise ValueError("[ERROR][ModelHandler] 빈 DataFrame이 입력되었습니다.")
        if feature_cols is None:
            raise ValueError("[ERROR][ModelHandler] 학습에 사용할 feature_cols이 입력되지 않았습니다.")
        if target_col is None:
            raise ValueError("[ERROR][ModelHandler] 학습에 사용할 target_col이 입력되지 않았습니다.")
        if target_col not in df.columns:
            raise ValueError(f"[ERROR][ModelHandler] target_col({target_col})이 df에 없습니다.")
        
        self.df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.base_model = base_model.lower()
        self.base_params = base_parmas
        self.seed = seed
        self.n_splits = n_splits
        self.verbose = verbose
        
        self.n_classes = n_classes if n_classes is not None else self.df[target_col].nunique()
        self.model = self.__build_models__()
    
    def __build_models__(self):
        models = ["lgb", "lgbm", "lightgbm", "cat", "catboost", "xgb", "xgboost"]
        if self.base_model == "lgb" or self.base_model == "lgbm" or self.base_model == "lightgbm":
            self.model = LGBMClassifier(**self.base_params)
        elif self.base_model == 'cat' or self.base_model == "catboost":
            self.model = CatBoostClassifier(**self.base_params)
        elif self.base_model == "xgb" or self.base_model == "xgboost":
            self.model = XGBClassifier(**self.base_params)
        else:
            raise ValueError(f"[ERROR][ModelHandler] base 모델명은 다음과 같은 형식을 지원합니다\n{models}")
    
    def __train__(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        '''내부 학습/검증 루프'''
        train_X, train_y = train_df[self.feature_cols].values, train_df[self.target_col].values
        val_X, val_y = val_df[self.feature_cols].values, val_df[self.target_col].values
        self.model.fit(train_X, train_y, eval_set=[(val_X,val_y)], verbose=self.verbose)
        proba = self.model.predict_proba(val_X)
        pred = proba.argmax(1)
        return proba, pred
    
    def add_TabNet_feature(self,):
        '''TabNet을 사용해 DataFrame에 확률 피쳐를 추가합니다'''
        pass
    
    def train_all(self):
        '''검증 없이 모든 데이터 학습'''
        X,y = self.df[self.feature_cols].values, self.df[self.target_col].values
        self.model.fit(X,y)
        pass
    
    def train_cv(self,):
        '''K-Fold 검증 학습'''
        pass
    
    