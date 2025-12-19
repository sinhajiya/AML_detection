from preprocess_graphcreation import load_transactions, build_temporal_graph
from preprocess_motifs import motifs

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import onnxruntime as ort
import numpy as np
import pandas as pd
import os
import joblib

dataset_path = "E:\codes\AML_detection\AlmnetFraudDataset\AMLNet_August 2025.csv"
onnx_path = "E:\codes\AML_detection\model\isolation_forest.onnx"

df = load_transactions(
    csv_path=dataset_path,
)

df = df.sort_values("timestamp")
n = len(df)
train_df = df.iloc[: int(0.6 * n)]
test_df  = df.iloc[int(0.6 * n):]

print("Building train graph...")
out_tr, in_tr = build_temporal_graph(
    train_df,
    # src_col="nameOrig",
    # dst_col="nameDest",
    # amount_col="amount"
)

print("Extracting train motifs...")
feat_tr = motifs(out_tr, in_tr)


features = feat_tr  
X = features.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination=0.002,   # ~0.2% anomalies (close to AMLNet)
    random_state=42,
    n_jobs=-1
)

iso.fit(X_scaled)

features["risk_score"] = -iso.decision_function(X_scaled)

features["is_anomaly"] = iso.predict(X_scaled) == -1

top_risky = features.sort_values("is_anomaly", ascending=False)

print(top_risky.head(10))

MODEL_DIR = "models/IsolationForestModel"

os.makedirs(MODEL_DIR, exist_ok=True)
model = iso

sklearn_model_path = os.path.join(MODEL_DIR, "isolation_forest.pkl")
joblib.dump(model, sklearn_model_path)

print("Sklearn model saved at:", sklearn_model_path)

test_out_dr, test_in_dr=build_temporal_graph(test_df)
test_features = motifs( test_out_dr, test_in_dr)
X_test_ = test_features.values
X_test = scaler.fit_transform(X_test_)


X_sample = X_test
n_features = X_sample.shape[1]

initial_type = [("input", FloatTensorType([None, n_features]))]

# onnx_model = convert_sklearn(
#     model,
#     initial_types=initial_type,
#     target_opset=12
# )

onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    target_opset={
        "": 12,               # standard ONNX ops
        "ai.onnx.ml": 3       # ML ops (IsolationForest)
    }
)


onnx_model_path = os.path.join(MODEL_DIR, "isolation_forest.onnx")

with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved at:", onnx_model_path)


sk_pred = model.predict(X_sample)
sk_score = model.decision_function(X_sample)



session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

onnx_pred = session.run(None, {input_name: X_sample.astype(np.float32)})[0]



print("Sklearn prediction:", sk_pred[:10])
print("ONNX prediction   :", onnx_pred[:10])

onnx_pred = onnx_pred.ravel()   

assert np.allclose(sk_pred, onnx_pred), " ONNX and sklearn outputs differ"
print("ONNX and sklearn inference matches..")

