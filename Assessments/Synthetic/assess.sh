for dataset in MATH WildChat10K; do
    for rate in 0.2 0.4 0.5; do
        for size in {1..20}; do
            python -m AssociatedInstances.annotate --dataset ${dataset} --split full --capability_path "Datasets/${dataset}/eval_results/synthetic/[base=0.7]_[drate=${rate}]_[seed=0]/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/[split=full]weakness-profiles_[size=${size}].json"
            python -m Assessments.Synthetic.assess --dataset ${dataset} --results_path [base=0.7]_[drate=${rate}]_[seed=0] \
                --method TextDiff --predictor "[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/[split=full]weakness-profiles_[size={PLACEHOLDER}]" \
                --size ${size}
            
            python -m AssociatedInstances.annotate --dataset ${dataset} --split full --capability_path "Datasets/${dataset}/eval_results/synthetic/[base=0.7]_[drate=${rate}]_[seed=0]/QualEval/[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=full]/weakness-profiles_[size=${size}].json"
            python -m Assessments.Synthetic.assess --dataset ${dataset} --results_path [base=0.7]_[drate=${rate}]_[seed=0] \
                --method QualEval --predictor "[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=full]/weakness-profiles_[size={PLACEHOLDER}]" \
                --size ${size}
            
            python -m AssociatedInstances.annotate --dataset ${dataset} --split full --capability_path "Datasets/${dataset}/eval_results/synthetic/[base=0.7]_[drate=${rate}]_[seed=0]/EvalTree/TREE=[stage3-RecursiveClustering]_[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles_varying_threshold/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[size=${size}].json"
            python -m Assessments.Synthetic.assess --dataset ${dataset} --results_path [base=0.7]_[drate=${rate}]_[seed=0] \
                --method EvalTree --predictor "TREE=[stage3-RecursiveClustering]_[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles_varying_threshold/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[size={PLACEHOLDER}]" \
                --size ${size}
        done
    done
done