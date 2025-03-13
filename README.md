[This](https://zhiyuan-zeng.github.io/EvalTree) is a demo of **capability trees** constructed by the **EvalTree** method proposed in [*EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees*](https://arxiv.org/abs/2503.08893), implemented by [Zhiyuan Zeng](https://zhiyuan-zeng.github.io) using [OpenHands](https://app.all-hands.dev).

## 🌳 Tree Structure

- Each **node** represents a specific **capability** described in natural language and is linked to a subset of benchmark instances that evaluate this capability.
- The **root node** is linked to all instances in the benchmark.
- Each **non-leaf node** partitions its linked instances into **subsets**, corresponding to more specific **sub-capabilities** represented by its child nodes.
- Each **leaf node** corresponds one-to-one with an individual **instance**. Instances linked to a node are exactly the leaves in its subtree.
- Model performance is evaluated at each node (more specifically, instances linked to the node).

## 🖱️ Interacting with the Capability Tree

- **Left-click** a **non-leaf node** to **expand/collapse** it.
- **Right-click** a **non-leaf node** to view **detailed information**, including: (1) the number of instances linked to each child node, along with each child node's basic information; (2) all benchmark instances linked to the node; (3) the ranking of different models on this node.

## 🎨 Understanding Information on Nodes

For **non-leaf nodes**:
- **Distinction focus** (how a node differs from its siblings) is highlighted in <span style="color:#D9A441;">●</span>.
- **Capability descriptions** are displayed in <span style="color:#FFFFFF; background-color:#4575B4;">●</span>, which correspond to the "capability descriptions" in the paper.

For **leaf nodes**:
- The corresponding **instance input** is displayed on the node.

## 🔍 Navigation and Search

- **Move the tree** by **dragging** the interface.
- **Zoom in/out** by **scrolling up/down**.
- Use the **search box** (top-left) to find specific nodes.

## 🔄 Switching Benchmarks and Models

- Click the **benchmark-switching button** to switch between different benchmarks.
- You can also switch between different models' capability trees. Press the **[NONE]** button to view the capability tree without specifying a particular model.
- When the benchmark is fixed, **all models have capability trees with the same hierarchical structure and capability descriptions**, but their performance at each node differs.

## 📊 Model Performance

- On **MATH, MMLU, and DS-1000**, each model is evaluated on each benchmark instance and given a **binary correctness score** (i.e., 0/1). The **accuracy** at a node is computed as the proportion of correct instances among all instances at that node.
- On **WildChat**, we use **LM-as-a-judge** to compare responses produced by **Llama 3.2 3B Instruct** and **Gemma 2 IT 2B**. The evaluation metric is **win-rate**, representing the percentage of instances where the model's response is preferred.
- On **Chatbot Arena**, we compute **Elo scores** of LMs based on human comparison pairs for instances linked to each node. If the number of comparison pairs is insufficient, the Elo rating computation does not converge, and we do not display rankings in such cases.
