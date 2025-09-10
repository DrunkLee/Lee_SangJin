from preprocessor import DataPreprocessor
from models import ModelHandler
import pandas as pd

train_path = "./data/train.csv"
test_path = "./data/test.csv"
sub_path = "./data/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sub_path)

pre = DataPreprocessor()
pre.fit_encoder(train_df)

train_processed = pre.transform(train_df, is_train = True)
test_processed = pre.transform(test_df, is_train = False)

feature_cols = [col for col in train_processed.columns if col not in ["ID", "support_needs"]]

print("="*50)
print(f"[INFO] 학습에 사용할 Feature 갯수: {len(feature_cols)}")
print("="*50)
target_col = "support_needs"

is_test = False

if is_test:
    model = ModelHandler(seed=42, save_dir="./train_result")
    model.fit_all(train_processed, feature_cols, target_col)
    model.predict(test_processed, sub_df, feature_cols)
else:
    model = ModelHandler(seed=42)
    model.fit_kfold(train_processed, feature_cols, target_col)
    model.fit_kfold_smoteenn(train_processed, feature_cols, target_col)