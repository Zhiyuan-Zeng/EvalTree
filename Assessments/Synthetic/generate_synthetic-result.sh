for dataset in MATH WildChat10K; do
    python -m AssociatedInstances.annotate --dataset ${dataset} --split full --capability_path Datasets/${dataset}/eval_results/synthetic/ground-truth.json
    for rate in 0.2 0.4 0.5; do
        python -m Assessments.Synthetic.generate_synthetic-result --dataset ${dataset} --prob_drate $rate
    done
done
