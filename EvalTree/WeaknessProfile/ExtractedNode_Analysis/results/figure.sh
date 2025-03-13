for dataset in MATH MMLU DS-1000; do
    python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.results.figure \
        --predictor_dataset $dataset --target_dataset $dataset
done

python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.results.figure \
    --predictor_dataset MATH --target_dataset CollegeMath
