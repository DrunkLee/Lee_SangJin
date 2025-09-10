import pandas as pd
import pickle
import os

class DataLoader:
    def __init__(self, path:str, used_features:list = None) -> pd.DataFrame:
        if not path:
            raise ValueError("[ERROR][DataLoader] DataFrame 생성을 위한 CSV 경로가 입력되지 않았습니다") 
        
        ext = os.path.splitext(path)[-1].lower()
        
        if ext not in [".csv",".pickle",".pkl"]:
            raise ValueError(f"[ERROR][DataLoader] 잘못된 파일 형식입니다.  CSV 또는 Pickle 파일만 허용됩니다: {ext}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR][DataLoader] 경로에 파일이 존재하지 않습니다: {path}")
        
        self.path = path
        self.ext = ext
        self.used_features = used_features
        
    def __load_df__(self) -> pd.DataFrame:
        if self.ext ==".csv":
            if self.used_features:
                return pd.read_csv(self.path, usecols=self.used_features).drop(columns=["ID"], errors="ignore")
            else:
                return pd.read_csv(self.path).drop(columns=["ID"], errors="ignore")
        elif self.ext in [".pickle",".pkl"]:
            with open(self.path, "rb") as f:
                df = pickle.load(f)
            if self.used_features:
                df = df[self.used_features]
            if "ID" in df.columns:
                df = df.drop(columns=["ID"])
            return df
        
    def save_pickle(self, df:pd.DataFrame, save_dir:str):
        if not df:
            raise ValueError("[ERROR][DataLoader] 데이터프레임이 입력되지 않았습니다.")
        if not save_dir:
            raise ValueError("[ERROR][DataLoader] save_dir이 입력되지 않았습니다.")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "train.pickle")
        with open(save_path, "wb") as f:
            pickle.dump(df, f)
        print(f"[INFO][DataLoader] DataFrame이 Pickle로 저장되었습니다: {save_path}")
        
    def __call__(self, *args, **kwds) -> pd.DataFrame:
        return self.__load_df__()