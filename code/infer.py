import numpy as np
import pandas as pd
import onnxruntime as ort

from preprocess_graphcreation import load_transactions
from sklearn.metrics import average_precision_score, roc_auc_score

onnx_model_path = r'isolation_forest.onnx'
feat_train_path = r"AMLNet_Motifs\AMLNet_Motifs_singlewindow\feat_train.parquet"
feat_test_path  = r"AMLNet_Motifs\AMLNet_Motifs_singlewindow\feat_test.parquet"
dataset_path = r"AlmnetFraudDataset\AMLNet_August 2025.csv"

print("loading dataset")
df = load_transactions(csv_path=dataset_path)

df = df.sort_values("timestamp")
n = len(df)
print("loading train/test split")
train_df = df.iloc[: int(0.6 * n)]
test_df  = df.iloc[int(0.6 * n):]

feat_tr = pd.read_parquet(feat_train_path)
feat_te = pd.read_parquet(feat_test_path)

feat_te = feat_te.reindex(
    columns=feat_tr.columns,
    fill_value=0
)

test_features = feat_te.copy()

print("Running ONNX model inference")
sess = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name
print("ONNX input:", input_name)
print("ONNX outputs:", sess.get_outputs())

X_test = test_features.values.astype(np.float32)

outputs = sess.run(
    None,
    {input_name: X_test}
)

labels_onnx = outputs[0]   # +1 normal, -1 anomaly
scores_onnx = outputs[1]   # lower = more anomalous

test_features["risk_score"] = -scores_onnx
test_features["is_anomaly"] = (labels_onnx == -1)

def account_labels(df):
    return (
        df.groupby("nameOrig")["edge_features"]
          .apply(lambda s: any(
              d.get("isMoneyLaundering", 0) == 1 for d in s
          ))
          .astype(int)
    )

y_test_all = account_labels(test_df)

# ALIGN labels to scored accounts
y_test = y_test_all.reindex(
    test_features.index,
    fill_value=0
)

print("Accounts with labels:", len(y_test_all))
print("Accounts with scores:", len(test_features))
print("Positive accounts in scored set:", y_test.sum())

assert len(y_test) == len(test_features)

pr_auc = average_precision_score(
    y_test,
    test_features["risk_score"]
)

roc_auc = roc_auc_score(
    y_test,
    test_features["risk_score"]
)

print("TEST PR-AUC:", pr_auc)
print("TEST ROC-AUC:", roc_auc)


def recall_at_k(y_true, scores, k_frac):
    k = max(1, int(len(scores) * k_frac))
    idx = np.argsort(scores)[-k:]
    return y_true.iloc[idx].sum() / y_true.sum()

for k in [0.001, 0.005, 0.01, 0.02]:
    print(
        f"Recall@{k*100:.1f}%:",
        recall_at_k(y_test, test_features["risk_score"], k)
    )

top_risky = test_features.sort_values(
    "risk_score",
    ascending=False
)

print(top_risky.head(10))
