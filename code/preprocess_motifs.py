import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

TIME_WINDOW = timedelta(hours=1)     # motif window
MIN_FAN = 3                          # fan-in / fan-out threshold

def extract_motifs(out_edges, in_edges, window):
    chain = defaultdict(int)
    fanout = defaultdict(int)
    fanin = defaultdict(int)
    cycle = defaultdict(int)
    repeat = defaultdict(int)

    # ---------- CHAIN: A -> B -> C ----------
    for A in out_edges:
        for B, t1, _ in out_edges[A]:
            for C, t2, _ in out_edges.get(B, []):
                if t2 <= t1:
                    continue
                if t2 - t1 > window:
                    break
                chain[A] += 1

    # ---------- FAN-OUT: A -> many ----------
    for A, edges in out_edges.items():
        edges = sorted(edges, key=lambda x: x[1])
        for i in range(len(edges)):
            t0 = edges[i][1]
            receivers = set()
            for j in range(i, len(edges)):
                if edges[j][1] - t0 > window:
                    break
                receivers.add(edges[j][0])
            if len(receivers) >= MIN_FAN:
                fanout[A] += 1

    # ---------- FAN-IN: many -> A ----------
    for A, edges in in_edges.items():
        edges = sorted(edges, key=lambda x: x[1])
        for i in range(len(edges)):
            t0 = edges[i][1]
            senders = set()
            for j in range(i, len(edges)):
                if edges[j][1] - t0 > window:
                    break
                senders.add(edges[j][0])
            if len(senders) >= MIN_FAN:
                fanin[A] += 1

    # ---------- CYCLE: A -> B -> C -> A ----------
    for A in out_edges:
        for B, t1, _ in out_edges[A]:
            for C, t2, _ in out_edges.get(B, []):
                if t2 <= t1 or t2 - t1 > window:
                    continue
                for A2, t3, _ in out_edges.get(C, []):
                    if A2 == A and t3 > t2 and t3 - t1 <= window:
                        cycle[A] += 1

    # ---------- REPEAT: A -> B repeatedly ----------
    for A, edges in out_edges.items():
        edges = sorted(edges, key=lambda x: (x[0], x[1]))
        for i in range(len(edges) - 1):
            B1, t1, _ = edges[i]
            B2, t2, _ = edges[i + 1]
            if B1 == B2 and (t2 - t1) <= window:
                repeat[A] += 1

    return chain, fanout, fanin, cycle, repeat


def feature_table(chain, fanout, fanin, cycle, repeat):
    features = pd.DataFrame({
        "chain": chain,
        "fanout": fanout,
        "fanin": fanin,
        "cycle": cycle,
        "repeat": repeat
    }).fillna(0)

    return features


def motifs(out_edges, in_edges,
                       src_col="source",
                       dst_col="target",
                       time_col="timestamp",
                       amount_col=None):


    motifs = extract_motifs(out_edges, in_edges, TIME_WINDOW)
    features = feature_table(*motifs)

    return features


