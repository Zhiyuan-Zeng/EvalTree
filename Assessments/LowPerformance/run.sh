# MATH
models=("Llama-3.1-8B-Instruct" "dart-math-llama3-8b-uniform")
for model in "${models[@]}"; do
    bash Baselines/TextDiff/profile-generation.sh MATH real/${model} 4k-1k

    python -m Baselines.QualEval.WeaknessProfile.profile-generation \
        --dataset MATH --round 5 \
        --results_path real/${model} \
        --split 4k-1k

    python -m EvalTree.WeaknessProfile.profile-generation_varying-threshold \
        --dataset MATH \
        --tree_path stage3-RecursiveClustering/[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --results_path real/${model}
done





# WildChat10K
bash Baselines/TextDiff/profile-generation.sh WildChat10K real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it] 8k-2k

python -m Baselines.QualEval.WeaknessProfile.profile-generation \
        --dataset WildChat10K --round 5 \
        --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it] \
        --split 8k-2k

python -m EvalTree.WeaknessProfile.profile-generation_varying-threshold \
        --dataset WildChat10K \
        --tree_path stage3-RecursiveClustering/[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]
