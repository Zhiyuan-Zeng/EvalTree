python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset MATH \
    --tree_path stage3-RecursiveClustering/[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset MATH \
    --embedding_split [exclusion]4k-1k

python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset MMLU \
    --tree_path stage3-RecursiveClustering/[split=10042-4000]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset MMLU \
    --embedding_split [exclusion]10042-4000

python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset DS-1000 \
    --tree_path stage3-RecursiveClustering/[split=600-400]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset DS-1000 \
    --embedding_split [exclusion]600-400

python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset WildChat10K \
    --tree_path stage3-RecursiveClustering/[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset WildChat10K \
    --embedding_split [exclusion]8k-2k



python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset MATH \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset CollegeMath \
    --embedding_split full

python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset WildChat10K \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset ShareGPT10K \
    --embedding_split full
python -m EvalTree.stage3-RecursiveClustering.locate \
    --tree_dataset WildChat10K \
    --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
    --embedding_dataset Chatbot-Arena \
    --embedding_split full
