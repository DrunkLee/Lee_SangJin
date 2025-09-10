import pandas as pd
    
class Preprocessor:
    def __init__(self, df:pd.DataFrame, is_train:bool = True) -> pd.DataFrame:
        if df is None:
            raise ValueError("[ERROR][Preprocessor] DataFrame이 입력되지 않았습니다 (None).")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"[ERROR][Preprocessor] 입력값이 pandas의 DataFrame이 아닙니다: {type(df)}")
        if df.empty:
            raise ValueError("[ERROR][Preprocessor] 빈 DataFrame이 입력되었습니다.")
        self.df = df.copy()
        self.is_train = is_train
    
    def fit(self):
        pass
    
    def transform(self):
        pass
    
    def fit_transform(self) -> pd.DataFrame:
        self.fit()
        return self.df