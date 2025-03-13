for dataset in MATH DS-1000; do
    for config in Generic-Capability TextDiff QualEval EvalTree Directly-Sampled; do
        for seed in {0..4}; do
            python -m Assessments.Extrinsic.data.generate_data.generate_data \
                --dataset ${dataset} \
                --config ${config} \
                --seed ${seed}
        done
    done
done
