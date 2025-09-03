from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split as tts
import pandas as pd

def print_score(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print("\n" + "="*40)
    print(f"✅ {model_name} 평가 결과")
    print("="*40)
    print(f"  - 정확도 (Accuracy)  : {acc:.4f}")
    print(f"  - 정밀도 (Precision) : {pre:.4f}")
    print(f"  - 재현율 (Recall)    : {rec:.4f}")
    print(f"  - F1 점수 (F1 Score) : {f1:.4f}")
    print("="*40)

train_path = "./data/train.csv"
train_df = pd.read_csv(train_path)
train_df['gender'] = train_df['gender'].map({'F':0, "M": 1})
train_df['subscription_type'] = train_df['subscription_type'].map({'member' : 0, 'plus' : 1, 'vip' : 2})

target_col = "support_needs"
ex_col = ["ID"] + [target_col]
feature_col = [col for col in train_df.columns if col not in ex_col]

train,val = tts(train_df, test_size=0.2, random_state=42, stratify=train_df[target_col])
print(f"학습: {len(train)}, 검증: {len(val)}")

X_train, y_train = train[feature_col], train[target_col]
X_val, y_val = val[feature_col], val[target_col]

print("\n🚀 XGBoost 모델을 학습합니다...")
xgb_model = XGBClassifier(random_state=42, n_estimators=100)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_val)
print_score("XGBoost", y_val, xgb_pred)

print("\n🚀 LightGBM 모델을 학습합니다...")
lgbm_model = lgb.LGBMClassifier(random_state=42, n_estimators=100, class_weight="balanced")
lgbm_model.fit(X_train, y_train)
lgbm_pred = lgbm_model.predict(X_val)
print_score("LightGBM", y_val, lgbm_pred)