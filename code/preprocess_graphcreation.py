# graphcreation.py

import pandas as pd
from collections import defaultdict
from datetime import datetime

# --------------------------------------------------
# LOAD AMLNET TRANSACTIONS (KEEP LABELS)
# --------------------------------------------------

def load_transactions(
    csv_path="/content/drive/MyDrive/AlmnetFraudDataset/AMLNet_August 2025.csv",
    src_col="nameOrig",
    dst_col="nameDest",
    step_col="step",
    amount_col="amount"
):
    

    df = pd.read_csv(csv_path)

    base_date = datetime(2024, 1, 1)
    df["timestamp"] = base_date + pd.to_timedelta(df[step_col], unit="h")

    df["edge_features"] = df.apply(
        lambda r: {
            "amount": r[amount_col],

            "isMoneyLaundering": r.get("isMoneyLaundering", None),
            "isFraud": r.get("isFraud", None),
            "laundering_typology": r.get("laundering_typology", None),
            "fraud_probability": r.get("fraud_probability", None),

            "type": r.get("type", None),
            "category": r.get("category", None),
            "oldbalanceOrg": r.get("oldbalanceOrg", None),
            "newbalanceOrig": r.get("newbalanceOrig", None),

            "hour": r.get("hour", None),
            "day_of_week": r.get("day_of_week", None),
            "day_of_month": r.get("day_of_month", None),
            "month": r.get("month", None),
        },
        axis=1
    )

    df = df.sort_values("timestamp")

    return df[[src_col, dst_col, "timestamp", "edge_features"]]


def build_temporal_graph(
    df,
    src_col="nameOrig",
    dst_col="nameDest",
    time_col="timestamp",
    edge_feat_col="edge_features"
):
    

    out_edges = defaultdict(list)
    in_edges = defaultdict(list)

    for row in df.itertuples(index=False):
        src = getattr(row, src_col)
        dst = getattr(row, dst_col)
        ts  = getattr(row, time_col)
        ef  = getattr(row, edge_feat_col)

        out_edges[src].append((dst, ts, ef))
        in_edges[dst].append((src, ts, ef))

    return out_edges, in_edges
