# EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees

This repository contains the **code and data** for the paper:

**EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees**
Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, Pang Wei Koh
*Preprint.*

## 🔗 Resources
- 📄 **[Paper](https://arxiv.org/abs/2503.08893)**
- 💾 **[Code & Data](https://github.com/Zhiyuan-Zeng/EvalTree)**
- 🌐 **[Web Interface (Interactive Demo of Capability Trees)](https://zhiyuan-zeng.github.io/EvalTree)**

If you find our work useful, please consider citing:

```bibtex
@article{zeng2025evaltree,
  title={EvalTree: Profiling Language Model Weaknesses via Hierarchical Capability Trees},
  author={Zeng, Zhiyuan and Wang, Yizhong and Hajishirzi, Hannaneh and Koh, Pang Wei},
  journal={arXiv preprint 2503.08893},
  year={2025}
}
```

## Bug Reports & Questions

If you have any questions about the code or the paper, feel free to contact [Zhiyuan Zeng](https://zhiyuan-zeng.github.io/) (`zhiyuan1zeng@gmail.com` or `zyzeng@cs.washington.edu`).

If you encounter any issues while using the code or want to report a bug, please open an issue. When reporting a problem, provide detailed information so we can assist you more effectively.

## Setup

### Installation and Configuration

Install the required dependencies.
```bash
pip install -r requirements.txt
```

Set up your keys and ensure that your OpenAI and Hugging Face credentials are correctly configured before running the code.
```bash
export OPENAI_API_KEY="your_openai_api_key"
export HF_TOKEN="your_huggingface_access_token"
```

### Model Evaluation Results

A model's evaluation result on a benchmark is stored in
`Datasets/BENCHMARK/eval_results/real/MODEL/results.json`.
This is the **performance metric vector**, as all weakness profiling methods operate on performance metrics rather than original generation results.

For **instruction-following benchmarks**, `MODEL` takes the form `[MODEL1]BEAT[MODEL2]`, indicating it is the preference label vector determined by LM-as-a-judge comparing `MODEL1` and `MODEL2`. In this case, `1` means `MODEL1` is preferred, and `2` means `MODEL2` is preferred.
Each instance has two preference labels to account for both the original order and the swapped response order in pairwise comparisons.

## Preparation of EvalTree and QualEval

### EvalTree

Using **EvalTree**, we first run the automatic four-stage tree construction pipeline on each benchmark.

#### Step 1: Capability Annotation
We first run the **Capability Annotation** stage. Precomputed results are available in `Datasets/BENCHMARK/EvalTree/stage1-CapabilityAnnotation/[annotation=gpt-4o-mini].json`.
```bash
bash EvalTree/stage1-CapabilityAnnotation/annotate.sh
# Precomputed results are already available.
```

#### Step 2: Capability Embedding
We then run the **Capability Embedding** stage. Outputs are stored in `Datasets/BENCHMARK/EvalTree/stage2-CapabilityEmbedding/[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small].bin`.
```bash
bash EvalTree/stage2-CapabilityEmbedding/embedding.sh
```

#### Step 3: Recursive Clustering-Based Construction
Next, we run the **Recursive Clustering-Based Construction** stage. Outputs are stored in `Datasets/BENCHMARK/EvalTree/stage3-RecursiveClustering/[split=SPLIT]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10].bin`.
```bash
bash EvalTree/stage3-RecursiveClustering/build.sh
```

#### Step 4: Capability Description
Finally, we run the **Capability Description** stage. Precomputed results are available in `Datasets/BENCHMARK/EvalTree/stage3-RecursiveClustering/[split=SPLIT]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]_[stage4-CapabilityDescription-model=gpt-4o-mini].json`.
```bash
bash EvalTree/stage4-CapabilityDescription/describe.sh
# Precomputed results are already available.
```

#### Confidence Interval Computation
In our experiments, we do not explicitly construct capability trees (i.e., the tree structure with a model’s performance computed at each node).
Instead, we directly compute the **confidence interval of the binomial test** for each model's evaluation result.
This allows us to efficiently generate weakness profiles at varying threshold $\tau$.
Precomputed results are available in `Datasets/BENCHMARK/eval_results/real/MODEL/EvalTree/TREE=[stage3-RecursiveClustering]_[split=SPLIT]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/confidence_interval.json`.
```bash
bash EvalTree/WeaknessProfile/confidence_interval.sh
# Precomputed results are already available.
```

### QualEval

We run the following commands to obtain the one-level capability categorization structure of [QualEval](https://arxiv.org/abs/2311.02807) for each benchmark.
```bash
bash Baselines/QualEval/stage1-CapabilityDiscovery/discover.sh
# Precomputed results are already available.

bash Baselines/QualEval/stage2-CapabilityAssignment/assign.sh
# Precomputed results are already available.
```

## Assessments for Weakness Profiling Methods

### Low-Performance Identification Assessment

We first run all weakness profiling methods on all evaluation results.
As described in the paper, for each method, we tune its $\tau$ to generate all possible weakness profiles $\{W_{\tau_1}, W_{\tau_2}, \dots\}$.
```bash
bash Assessments/LowPerformance/run.sh
# Precomputed results are already available.
```

We then assess all methods using Low-Performance Identification Assessment.
The assessment results are stored in `Assessments/LowPerformance/results/BENCHMARK/real/MODEL`.
`size2val1` and `num2val2` correspond to the results of $\min\{\sum_{w_i \in W_{\tau}} F(A(w_i)) / |W_{\tau}| \mid \forall {\tau}, |W_{\tau}| \geq M'\}$ and $\min\{F(S_{\tau}) \mid \forall {\tau}, |S_{\tau}| \geq N'\}$ in the paper, respectively.
```bash
bash Assessments/LowPerformance/assess.sh
# Precomputed results are already available.
```

Finally, we generate the result figure.
```bash
python -m Assessments.LowPerformance.results.figure
# The figure is already available.
```

### Ground-Truth Weakness Assessment

As preparation for Ground-Truth Weakness Assessment, we manually curated 10 ground-truth weaknesses at various granularities for MATH and WildChat10K, respectively.
The ground-truth weakness profile is stored in `Datasets/{MATH, WildChat10K}/eval_results/synthetic/ground-truth.json`.

For each benchmark, we first generate three synthetic evaluation results using the hyperparameters $p=0.7$ and $d \in \{0.2, 0.4, 0.5\}$.
```bash
bash Assessments/Synthetic/generate_synthetic-result.sh
# Precomputed results are already available.
```

We then run all weakness profiling methods on all synthetic evaluation results.
```bash
bash Assessments/Synthetic/run.sh
# Precomputed results are already available.
```

Finally, we assess all methods using Ground-Truth Weakness Assessment.
The assessment results are stored in `Assessments/Synthetic/results/{MATH, WildChat10K}/[base=0.7]_[drate={0.2, 0.4, 0.5}]_[seed=0]`.
```bash
bash Assessments/Synthetic/assess.sh
# Precomputed results are already available.
```

Finally, we generate the result figure.
```bash
python -m Assessments.Synthetic.results.figure --metrics F1
python -m Assessments.Synthetic.results.figure --metrics Precision
python -m Assessments.Synthetic.results.figure --metrics Recall
# The figures are already available.
```

### Extrinsic Assessment: Weakness-Guided Training Data Collection

We first generate weakness profiles using all weakness profiling methods.
```bash
bash Assessments/Extrinsic/data/profile-generation.sh
# Precomputed results are already available.
```

We then generate synthetic data inputs.
For each synthetic data collection strategy, we construct a pool of synthetic data inputs.
When experimenting with different seeds, we sample from this pool to simulate re-running synthetic data generation with a new seed.
```bash
bash Assessments/Extrinsic/data/generate_input.sh
# Precomputed results are already available.
```

We generate outputs for each input across all data collection strategies.
```bash
bash Assessments/Extrinsic/data/generate_output.sh
# Precomputed results are already available.
```

Next, we construct training sets for each data collection strategy using five different seeds.
```bash
bash Assessments/Extrinsic/data/generate_data/generate_data.sh
# Precomputed results are already available.
```

We then finetune the initial LM (Llama 3.1 8B Instruct for MATH and DeepSeek-Coder-Base 6.7B for DS-1000) on each training set.
Training outputs are stored in `../{MATH, DS-1000}_checkpoints/STRATEGY_[seed=SEED]_[epoch=EPOCH]`.
```bash
bash Assessments/Extrinsic/training/train.sh
```

Precomputed generation results and evaluation results are available in `Assessments/Extrinsic/results/{MATH, DS-1000}`.
Using the evaluation results, we generate the result figure.
```bash
python -m Assessments.Extrinsic.results.figure
# The figure is already available.
```

## Analysis on Threshold $\tau$ for Node Extraction

As described in the paper, we locate the position of each instance in the test set on the capability tree by first computing its capability embedding and then traversing from the root guided by it.
We precompute each test set instance's traversal trajectory on the capability tree.
```bash
bash EvalTree/WeaknessProfile/ExtractedNode_Analysis/locate.sh
# Precomputed results are already available.
```

We then compute the performance on weakness/strength instances as the threshold varies.
Precomputed results are available in `EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/BENCHMARK1->BENCHMARK2`.
```bash
bash EvalTree/WeaknessProfile/ExtractedNode_Analysis/analysis_varying-threshold.sh
# Precomputed results are already available.
```

Finally, we generate the result figure.
```bash
bash EvalTree/WeaknessProfile/ExtractedNode_Analysis/results/figure.sh
python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.results.figure_instruction-following
# The figures are already available.
```

## User Interface of Capability Trees

The `demo` branch contains code to help you build an interface for exploring capability trees interactively.
You can see a **[demo](https://zhiyuan-zeng.github.io/EvalTree)** of the interface.

Once you have constructed the tree for a benchmark (following the steps above) and added your own model evaluation results, proceed with the following steps:

1. **Generate Capability Distinctions**:  
   Run `EvalTree/EvalTree/stage5-CapabilityDistinguishing` to generate a **natural language distinction** for each (non-root) node. It differentiates each node from its siblings, giving a more concise and user-friendly description of its capability.

2. **Prepare Data for the Interface**:  
   Execute `EvalTree/build_data.py` to generate the necessary data files for the interface.

3. **Customize Metadata**:  
   Modify `meta.json` to include your **benchmark and model information**.

