for dataset in MATH WildChat10K; do
    for rate in 0.2 0.4 0.5; do
        bash Baselines/TextDiff/profile-generation.sh ${dataset} synthetic/[base=0.7]_[drate=${rate}]_[seed=0] full

        python -m Baselines.QualEval.WeaknessProfile.profile-generation \
            --dataset ${dataset} --round 5 \
            --results_path synthetic/[base=0.7]_[drate=${rate}]_[seed=0] \
            --split full

        python -m EvalTree.WeaknessProfile.confidence_interval \
            --dataset ${dataset} \
            --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --results_path synthetic/[base=0.7]_[drate=${rate}]_[seed=0]
        python -m EvalTree.WeaknessProfile.profile-generation_varying-threshold \
            --max_profile_size 20 \
            --dataset ${dataset} \
            --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --results_path synthetic/[base=0.7]_[drate=${rate}]_[seed=0]
    done
done
