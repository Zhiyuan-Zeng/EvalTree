python -m EvalTree.stage3-RecursiveClustering.build --dataset MATH --split 4k-1k
python -m EvalTree.stage3-RecursiveClustering.build --dataset MATH --split full

python -m EvalTree.stage3-RecursiveClustering.build --dataset WildChat10K --split 8k-2k
python -m EvalTree.stage3-RecursiveClustering.build --dataset WildChat10K --split full

python -m EvalTree.stage3-RecursiveClustering.build --dataset DS-1000 --split 600-400

python -m EvalTree.stage3-RecursiveClustering.build --dataset MMLU --split 10042-4000
