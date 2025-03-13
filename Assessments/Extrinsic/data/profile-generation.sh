# MATH
# Assume that we have already run `bash Assessments/LowPerformance/run.sh`
python -m Baselines.TextDiff.generate \
    --dataset MATH \
    --results_path real/Llama-3.1-8B-Instruct \
    --split 4k-1k \
    --output_instances 9

python -m Baselines.QualEval.WeaknessProfile.profile-generation \
    --dataset MATH --round 5 \
    --results_path real/Llama-3.1-8B-Instruct \
    --split 4k-1k \
    --output_instances 9

python -m EvalTree.WeaknessProfile.profile-generation \
    --dataset MATH \
    --tree_path stage3-RecursiveClustering/[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --results_path real/Llama-3.1-8B-Instruct \
    --threshold 0.4





# DS-1000
if [ ! -d "Datasets/DS-1000/eval_results/real/deepseek-coder-6.7b-base/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]" ]; then
    echo "Run compare.py"
    python -m Baselines.TextDiff.compare --dataset DS-1000 --results_path real/deepseek-coder-6.7b-base
else 
    echo "No need to run compare.py"
fi
python -m AssociatedInstances.annotate --dataset DS-1000 --split 600-400 --capability_path Datasets/DS-1000/eval_results/real/deepseek-coder-6.7b-base/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/weakness-profile.json
python -m Baselines.TextDiff.generate \
    --dataset DS-1000 \
    --results_path real/deepseek-coder-6.7b-base \
    --split 600-400 \
    --output_instances 5

python -m Baselines.QualEval.WeaknessProfile.profile-generation \
    --dataset DS-1000 --round 4 \
    --results_path real/deepseek-coder-6.7b-base \
    --split 600-400 \
    --output_instances 5

python -m EvalTree.WeaknessProfile.profile-generation \
    --dataset DS-1000 \
    --tree_path stage3-RecursiveClustering/[split=600-400]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --results_path real/deepseek-coder-6.7b-base \
    --threshold 0.4
