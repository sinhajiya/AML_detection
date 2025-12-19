# Problem Statement

Build an AI-based system that detects potential money laundering risks by learning from transactional
patterns, behavioral anomalies, and network relationships in financial data. The system must work
strictly on synthetic or fully anonymized simulated datasets, identify suspicious entities or flows,
and export the final trained model in ONNX format.

# Architecture overview

We had AMLnet dataset from which we create graph having nodes and edges along with timestap, from this graph we identifies motifs(observed patterns for any particular user) all these comes under preprocessing. After this preprocessing step we use Isolated classifier to calculate the score then based on this score, we use top k method to calculate recall.


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
python infer.py
```



# Additional Points

### Dataset: AMLNet[https://zenodo.org/records/16736515]

- Total transactions: 1,090,173 (1M+)
- Legitimate transactions: 1,088,428 (99.84%)
- Money laundering transactions: 1,745 (0.16%)

### Data preprocessing 

- files: code\graphcreation.py and code\motifs.py

Each transaction is modeled as a directed temporal edge in a transaction graph. Accounts correspond to nodes, and each transaction from an originating account to a destination account creates a directed edge annotated with a timestamp and transaction attributes.
Formally, the transaction graph is defined as
𝐺=(𝑉,𝐸), where: V represents customer accounts, 𝐸 represents transactions ordered by time.

### Motif extraction

For each account, we compute motif-based features by counting the occurrence of predefined temporal patterns, including:

Chain motifs (A → B → C),

Fan-out motifs (A → {B₁, B₂, …}),

Fan-in motifs ({B₁, B₂, …} → A),

Cycle motifs (A → B → C → A),

Repeated transfer motifs (A → B multiple times).

* Operates at the **account (node) level**
* Transactions are **aggregated into temporal motifs**

---

### Temporal Motif Features

* Extracted across **multiple transactions**
* Capture **higher-order behavioral patterns**, including:
  * Transaction chains (A → B → C)
  * Fan-in / fan-out structures
  * Cyclic money flows
  * Repeated transfers between accounts

---

### Isolation Forest

* Computes an **anomaly score** for each account:
  * Higher score → more anomalous
  * Lower score → more normal

---

### Contamination

* `contamination = c` specifies the **expected fraction of anomalies**
* The model:

  * Sorts anomaly scores
  * Selects the top **c × 100%** as anomalies
 

### Some metrics

- TEST PR-AUC: 0.028514915487905728
- TEST ROC-AUC: 0.556494823578419

- Recall@0.1%: 0.017921146953405017
- Recall@0.5%: 0.07526881720430108
- Recall@1.0%: 0.15053763440860216
- Recall@2.0%: 0.2903225806451613


### Top Flagged Accounts (Example Output)

| Account ID | Chain | Fan-out | Fan-in | Cycle | Repeat | Risk Score | is_anomaly |
|------------|-------|---------|--------|-------|--------|------------|---------|
| C9148 | 0 | 30 | 35 | 0 | 14 | 0.069262 | True |
| C4133 | 0 | 19 | 24 | 0 | 9  | 0.069262 | True |
| C711  | 0 | 24 | 29 | 0 | 9  | 0.069262 | True |
| C1307 | 0 | 24 | 52 | 0 | 6  | 0.069262 | True |
| C9626 | 0 | 21 | 38 | 0 | 9  | 0.069262 | True |
| C3998 | 0 | 80 | 19 | 0 | 45 | 0.069262 | True |
| C4388 | 1 | 72 | 39 | 0 | 45 | 0.069262 | True |
| C8762 | 1 | 139 | 41 | 0 | 84 | 0.069262 | True |
| C5677 | 3 | 47 | 31 | 0 | 22 | 0.069262 | True |
| C8004 | 2 | 48 | 28 | 0 | 21 | 0.069262 | True |
