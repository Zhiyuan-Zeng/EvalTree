for dataset in MATH DS-1000; do
    for config in Generic-Capability TextDiff QualEval EvalTree Directly-Sampled; do
        for seed in {0..4}; do
            bash Assessments/Extrinsic/training/training_scripts/${dataset}.sh ${config}_[seed=${seed}]
        done
    done
done
