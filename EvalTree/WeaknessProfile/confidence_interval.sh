models=("gpt-4o-mini-2024-07-18" "Llama-3.1-8B-Instruct" "dart-math-llama3-8b-uniform")
for model in "${models[@]}"; do
    python -m EvalTree.WeaknessProfile.confidence_interval \
        --dataset MATH \
        --tree_path stage3-RecursiveClustering/[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --results_path real/${model}
    python -m EvalTree.WeaknessProfile.confidence_interval \
        --dataset MATH \
        --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --results_path real/${model}
done

python -m EvalTree.WeaknessProfile.confidence_interval \
    --dataset WildChat10K \
    --tree_path stage3-RecursiveClustering/[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]
python -m EvalTree.WeaknessProfile.confidence_interval \
    --dataset WildChat10K \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]

models=("gpt-4o-2024-08-06" "gpt-3.5-turbo-0613" "deepseek-coder-6.7b-base")
for model in "${models[@]}"; do
    python -m EvalTree.WeaknessProfile.confidence_interval \
        --dataset DS-1000 \
        --tree_path stage3-RecursiveClustering/[split=600-400]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --results_path real/${model}
done

models=("gpt-4o-mini-2024-07-18" "Llama-3.1-8B-Instruct" "Llama-3.1-Tulu-3-8B")
for model in "${models[@]}"; do
    python -m EvalTree.WeaknessProfile.confidence_interval \
        --dataset MMLU \
        --tree_path stage3-RecursiveClustering/[split=10042-4000]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --results_path real/${model}
done