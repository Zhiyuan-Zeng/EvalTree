# MATH
models=("Llama-3.1-8B-Instruct" "dart-math-llama3-8b-uniform")
for model in "${models[@]}"; do
    python -m AssociatedInstances.annotate --dataset MATH --split [exclusion]4k-1k --capability_path "Datasets/MATH/eval_results/real/${model}/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/weakness-profile.json"
    python -m Assessments.LowPerformance.assess \
        --dataset MATH --results_path real/${model} --split [exclusion]4k-1k \
        --method TextDiff --predictor "[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/[split=4k-1k]weakness-profiles_[size={PLACEHOLDER}]"
    
    python -m AssociatedInstances.annotate --dataset MATH --split [exclusion]4k-1k --capability_path "Datasets/MATH/eval_results/real/${model}/QualEval/[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=4k-1k]/weakness-profiles_[size=20].json"
    python -m Assessments.LowPerformance.assess \
        --dataset MATH --results_path real/${model} --split [exclusion]4k-1k \
        --method QualEval --predictor "[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=4k-1k]/weakness-profiles_[size={PLACEHOLDER}]"
    
    for index in {1..120}; do
        python -m AssociatedInstances.annotate --dataset MATH --split [exclusion]4k-1k --capability_path "Datasets/MATH/eval_results/real/${model}/EvalTree/TREE=[stage3-RecursiveClustering]_[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles_varying_threshold/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[index=${index}].json"
    done
    python -m Assessments.LowPerformance.assess \
        --dataset MATH --results_path real/${model} --split [exclusion]4k-1k \
        --method EvalTree --predictor "TREE=[stage3-RecursiveClustering]_[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles_varying_threshold/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[index={PLACEHOLDER}]"
done





# WildChat10K
python -m AssociatedInstances.annotate --dataset WildChat10K --split [exclusion]8k-2k --capability_path "Datasets/WildChat10K/eval_results/real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/weakness-profile.json"
python -m Assessments.LowPerformance.assess \
    --dataset WildChat10K --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it] --split [exclusion]8k-2k \
    --method TextDiff --predictor "[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/[split=8k-2k]weakness-profiles_[size={PLACEHOLDER}]"

python -m AssociatedInstances.annotate --dataset WildChat10K --split [exclusion]8k-2k --capability_path "Datasets/WildChat10K/eval_results/real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]/QualEval/[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=8k-2k]/weakness-profiles_[size=20].json"
python -m Assessments.LowPerformance.assess \
    --dataset WildChat10K --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it] --split [exclusion]8k-2k \
    --method QualEval --predictor "[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=8k-2k]/weakness-profiles_[size={PLACEHOLDER}]"

for index in {1..120}; do
    python -m AssociatedInstances.annotate --dataset WildChat10K --split [exclusion]8k-2k --capability_path "Datasets/WildChat10K/eval_results/real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]/EvalTree/TREE=[stage3-RecursiveClustering]_[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles_varying_threshold/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[index=${index}].json"
done
python -m Assessments.LowPerformance.assess \
    --dataset WildChat10K --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it] --split [exclusion]8k-2k \
    --method EvalTree --predictor "TREE=[stage3-RecursiveClustering]_[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles_varying_threshold/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[index={PLACEHOLDER}]"