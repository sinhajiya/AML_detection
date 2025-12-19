# Problem Statement

# Architecture overview


# Env setup
```bash
conda env create -f aml_env.yml
```
# training
```bash
train.py
```

# inferencing
```bash
python test.py
```

# Metrics

TEST PR-AUC: 0.028514915487905728
TEST ROC-AUC: 0.556494823578419

Recall@0.1%: 0.017921146953405017
Recall@0.5%: 0.07526881720430108
Recall@1.0%: 0.15053763440860216
Recall@2.0%: 0.2903225806451613

       chain  fanout  fanin  cycle  repeat  risk_score  is_anomaly
C9148    0.0    30.0   35.0    0.0    14.0    0.069262        True
C4133    0.0    19.0   24.0    0.0     9.0    0.069262        True
C711     0.0    24.0   29.0    0.0     9.0    0.069262        True
C1307    0.0    24.0   52.0    0.0     6.0    0.069262        True
C9626    0.0    21.0   38.0    0.0     9.0    0.069262        True
C3998    0.0    80.0   19.0    0.0    45.0    0.069262        True
C4388    1.0    72.0   39.0    0.0    45.0    0.069262        True
C8762    1.0   139.0   41.0    0.0    84.0    0.069262        True
C5677    3.0    47.0   31.0    0.0    22.0    0.069262        True
C8004    2.0    48.0   28.0    0.0    21.0    0.069262        True