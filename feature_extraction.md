# Dataset: AMLNet[https://zenodo.org/records/16736515]

- Total transactions: 1,090,173 (1M+)
- Legitimate transactions: 1,088,428 (99.84%)
- Money laundering transactions: 1,745 (0.16%)

# Data preprocessing 

- files: code\graphcreation.py and code\motifs.py

Each transaction is modeled as a directed temporal edge in a transaction graph. Accounts correspond to nodes, and each transaction from an originating account to a destination account creates a directed edge annotated with a timestamp and transaction attributes.

Formally, the transaction graph is defined as

𝐺=(𝑉,𝐸), where: V represents customer accounts, 𝐸 represents transactions ordered by time.

# Motif extraction

For each account, we compute motif-based features by counting the occurrence of predefined temporal patterns, including:

Chain motifs (A → B → C),
Fan-out motifs (A → {B₁, B₂, …}),
Fan-in motifs ({B₁, B₂, …} → A),
Cycle motifs (A → B → C → A),
Repeated transfer motifs (A → B multiple times).


### Modeling Level

* Operates at the **account (node) level**
* Transactions are **aggregated into temporal motifs**
* The model scores **entities and flows**, not individual transaction rows
* Core question addressed:
  **“Is this account’s behavior suspicious over time?”**

---

### Temporal Motif Features

* Extracted across **multiple transactions**
* Capture **higher-order behavioral patterns**, including:

  * Transaction chains (A → B → C)
  * Fan-in / fan-out structures
  * Cyclic money flows
  * Repeated transfers between accounts
* These features capture **coordination and structuring behavior**, not isolated events

---

### Role of Isolation Forest

* Computes an **anomaly score** for each account:

  * Higher score → more anomalous
  * Lower score → more normal
* Used as a **ranking model**, not a hard classifier

---

### Meaning of Contamination

* `contamination = c` specifies the **expected fraction of anomalies**
* The model:

  * Sorts anomaly scores
  * Selects the top **c × 100%** as anomalies
  * Labels them as `-1`
  * Labels all others as normal (`+1`)
* This defines a **contamination-based cutoff**, not a business decision threshold

---

If you want, I can:

* compress this further into **one slide**
* rewrite it in **exam / viva style**
* or adapt it for a **Methods** or **System Overview** section
