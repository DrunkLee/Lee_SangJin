from preprocess import DataPreProcess
from prompt_analysis import PromptAnalysis
from datetime import datetime
import os

csv_path = "./sample.csv"
save_dir = "./"
pickle_filename = "df_pick.pickle"

pickle_path = os.path.join(save_dir, pickle_filename)

preprocessor = DataPreProcess()
df = None

if os.path.exists(pickle_path):
    print(f"[INFO] '{pickle_path}' 파일이 존재하여 데이터를 불러옵니다.")
    df = preprocessor.load_pickle(pickle_path)
else:
    print(f"[INFO] '{pickle_path}' 파일이 존재하지 않아 새로 생성합니다.")
    df = preprocessor.run_and_save(csv_path, save_dir)

analyzer = PromptAnalysis(df)
analyzer.prepare_analysis(prompt_col='user_prompt')

report = analyzer.analyze_by_category(top_n=20)

now = datetime.now()
timestamp = now.strftime('%Y%m%d_%H%M%S')
filename = f"analysis_report_{timestamp}.txt"

try:
    with open(filename, 'w', encoding='utf-8') as f:
        for category, details in report.items():
            print(f"\n========== 분석 기준: {category} ==========", file=f)
            for label, pos_details in details.items():
                if not pos_details:
                    print(f"  --- 라벨 [{label}] 데이터 없음 ---", file=f)
                    continue
                print(f"  --- 라벨 [{label}] ---", file=f)
                for feature, words in pos_details.items():
                    if not words: continue
                    word_list = [f"{word}({count})" for word, count in words]
                    print(f"    [{feature}] Top {len(words)}: {', '.join(word_list)}", file=f)
    print(f"\n✅ 분석 결과가 '{filename}' 파일로 성공적으로 저장되었습니다.")

except Exception as e:
    print(f"\n파일 저장 중 오류가 발생했습니다: {e}")
