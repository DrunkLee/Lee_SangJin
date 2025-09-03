import pandas as pd
import os
import pickle

class DataPreProcess:
    def __init__(self):
        pass
    
    def __load_and_process(self, df_path):
        origin_df = pd.read_csv(df_path)
        split_df = origin_df["output"].str.split(",", expand=True)
        split_df.columns = ["유형","극성","시제","명확성"]
        df = pd.concat([origin_df, split_df], axis=1)
        df = df.drop("output", axis=1)
        return df
    
    def __save_pickle(self, df: pd.DataFrame, save_dir:str):
        df = df.copy()
        full_path = os.path.join(save_dir, "df_pick.pickle")
        with open(full_path, "wb") as f:
            pickle.dump(df, f)
        print(f'[INFO] DataFrame이 {full_path}에 저장되었습니다.')

    def load_pickle(self, pickle_path:str):
        with open(pickle_path, 'rb') as f:
            df = pickle.load(f)
        print(f'[INFO] {pickle_path}에서 DataFrame을 불러왔습니다.')
        return df
    
    def run_and_save(self, df_path:str, save_dir:str) -> pd.DataFrame:
        df = self.__load_and_process(df_path)
        self.__save_pickle(df, save_dir)
        return df