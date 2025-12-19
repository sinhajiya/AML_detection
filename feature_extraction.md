# Dataset: AMLNet[https://zenodo.org/records/16736515]

- Total transactions: 1,090,173 (1M+)
- Legitimate transactions: 1,088,428 (99.84%)
- Money laundering transactions: 1,745 (0.16%)

# Data preprocessing 

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

We use these as input to the model. 