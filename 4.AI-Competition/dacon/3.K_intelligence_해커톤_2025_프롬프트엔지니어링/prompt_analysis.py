from collections import Counter,defaultdict
from konlpy.tag import Kkma
from tqdm import tqdm
import pandas as pd

class PromptAnalysis:
    def __init__(self, df:pd.DataFrame):
        self.df = df
        self.kkma = Kkma()
        self._is_prepared = False
        
    def prepare_analysis(self, prompt_col: str):
        print(f"'{prompt_col}' 컬럼의 형태소 분석을 시작합니다.")
        tqdm.pandas(desc="POS TAGGING")
        self.df['pos_tags'] = self.df[prompt_col].progress_apply(
            lambda text: self.kkma.pos(text) if isinstance(text, str) else []
        )
        self._is_prepared = True
        print("프롬프트 형태소 분석 완료")

    def analyze_by_category(self, top_n: int = 10) -> dict:
        if not self._is_prepared:
            raise Exception("분석이 준비되지 않았습니다. 'prepare_analysis()'를 먼저 실행해주세요.")

        category_columns = ["유형", "극성", "시제", "명확성"]
        full_report = defaultdict(lambda: defaultdict(dict))
        target_features = ['NNG','VA','VXV', 'VX','MAG', 'EFN','EFQ','EPT','ETD','NNB']
        
        for category in tqdm(category_columns, desc="Analyzing Categories"):
            grouped = self.df.groupby(category)
            for label, group in grouped:
                all_pos_tags = [tag for tags_list in group['pos_tags'] for tag in tags_list]
                # unique_pos_tags = sorted(list({tag for word, tag in all_pos_tags})) # 모든 형태소
                for feature in target_features:
                    feature_words = [
                        word for word, tag in all_pos_tags
                        if tag == feature and len(word) >= 1
                    ]
                    
                    if not feature_words: continue
                    
                    counter = Counter(feature_words)
                    full_report[category][label][feature] = counter.most_common(top_n)
        return full_report