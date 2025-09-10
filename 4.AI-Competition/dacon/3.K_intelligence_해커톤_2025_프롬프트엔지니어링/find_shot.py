import pandas as pd
import re
from collections import defaultdict

CSV_PATH   = "./sample.csv"
MAX_LEN    = 40
PER_LABEL  = 2
SEED       = 42
UNIQUE     = True

TYPE_LABELS   = ["사실형", "추론형", "대화형", "예측형"]
POL_LABELS    = ["긍정", "부정", "미정"]
TENSE_LABELS  = ["과거", "현재", "미래"]
CERT_LABELS   = ["확실", "불확실"]

def clean_text(s: str) -> str:
    s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip("“”\"'‘’")
    return s

def select_per_label(df, col, label_list, k, used_idx, seed):
    chosen_idxs = []
    taken_by_label = defaultdict(int)

    for lab in label_list:
        pool1 = df[(df[col] == lab) & (~df.index.isin(used_idx))]
        n1 = min(k, len(pool1))
        if n1 > 0:
            idxs1 = pool1.sample(n=n1, random_state=seed).index.tolist()
            chosen_idxs += idxs1
            taken_by_label[lab] += n1
            used_idx.update(idxs1)

        if taken_by_label[lab] < k and not UNIQUE:
            need = k - taken_by_label[lab]
            pool2 = df[(df[col] == lab)]
            add = min(need, len(pool2))
            if add > 0:
                idxs2 = pool2.sample(n=add, random_state=seed+1).index.tolist()
                chosen_idxs += idxs2
                taken_by_label[lab] += add
                if UNIQUE:
                    used_idx.update(idxs2)

    return chosen_idxs, dict(taken_by_label)

df = pd.read_csv(CSV_PATH)
df["text"] = df["user_prompt"].apply(clean_text)
df["len"]  = df["text"].str.len()

split_df = df["output"].str.split(",", expand=True)
split_df.columns = ["유형", "극성", "시제", "명확성"]
for c in split_df.columns:
    df[c] = split_df[c]

short = df[df["len"] <= MAX_LEN].copy()

used = set()
order_plan = [
    ("유형", TYPE_LABELS),
    ("극성", POL_LABELS),
    ("시제", TENSE_LABELS),
    ("명확성", CERT_LABELS),
]

all_idxs = []
stats = {}

for col, labs in order_plan:
    idxs, taken = select_per_label(short, col, labs, PER_LABEL, used, SEED)
    all_idxs += idxs
    stats[col] = taken

seen = set()
ordered_idxs = []
for i in all_idxs:
    if i not in seen:
        seen.add(i)
        ordered_idxs.append(i)

sel = short.loc[ordered_idxs].reset_index(drop=True)

lines = ["예시"]
for i, row in enumerate(sel.itertuples(index=False), start=1):
    lines.append(f"{i} 입력 {row.text}")
    lines.append(f"{i} 출력 {row.output}")

block_text = "\n".join(lines)
print(block_text)

print("\n--- 요약 ---")
print(f"총 샷 수: {len(sel)} (UNIQUE={UNIQUE}, MAX_LEN={MAX_LEN}, PER_LABEL={PER_LABEL})")

for col, labs in order_plan:
    vc = sel[col].value_counts()
    want = ", ".join([f"{lab}:{stats[col].get(lab,0)}/{PER_LABEL}" for lab in labs])
    print(f"[{col}] 목표(선택수/요청수): {want}")
    print(f"[{col}] 실제분포: \n{vc}\n")
