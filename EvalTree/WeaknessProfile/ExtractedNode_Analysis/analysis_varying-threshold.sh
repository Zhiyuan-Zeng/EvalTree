for direction in lower higher; do
    # MATH -> MATH
    for model in gpt-4o-mini-2024-07-18 Llama-3.1-8B-Instruct dart-math-llama3-8b-uniform; do
        python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.analysis_varying-threshold \
            --direction ${direction} \
            --tree_dataset MATH \
            --tree_path stage3-RecursiveClustering/[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --embedding_dataset MATH \
            --embedding_split [exclusion]4k-1k \
            --results_path real/${model}
    done

    # MMLU -> MMLU
    for model in gpt-4o-mini-2024-07-18 Llama-3.1-8B-Instruct Llama-3.1-Tulu-3-8B; do
        python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.analysis_varying-threshold \
            --direction ${direction} \
            --tree_dataset MMLU \
            --tree_path stage3-RecursiveClustering/[split=10042-4000]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --embedding_dataset MMLU \
            --embedding_split [exclusion]10042-4000 \
            --results_path real/${model}
    done

    # DS-1000 -> DS-1000
    for model in gpt-4o-2024-08-06 gpt-3.5-turbo-0613 deepseek-coder-6.7b-base; do
        python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.analysis_varying-threshold \
            --direction ${direction} \
            --tree_dataset DS-1000 \
            --tree_path stage3-RecursiveClustering/[split=600-400]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --embedding_dataset DS-1000 \
            --embedding_split [exclusion]600-400 \
            --results_path real/${model}
    done

    # WildChat10K -> WildChat10K
    python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.analysis_varying-threshold \
        --direction ${direction} \
        --tree_dataset WildChat10K \
        --tree_path stage3-RecursiveClustering/[split=8k-2k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
        --embedding_dataset WildChat10K \
        --embedding_split [exclusion]8k-2k \
        --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]
    
    # MATH -> CollegeMath
    for model in gpt-4o-mini-2024-07-18 Llama-3.1-8B-Instruct dart-math-llama3-8b-uniform; do
        python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.analysis_varying-threshold \
            --direction ${direction} \
            --tree_dataset MATH \
            --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --embedding_dataset CollegeMath \
            --embedding_split full \
            --results_path real/${model}
    done

    # WildChat10K -> ShareGPT10K, Chatbot-Arena
    for dataset in ShareGPT10K Chatbot-Arena; do
        python -m EvalTree.WeaknessProfile.ExtractedNode_Analysis.analysis_varying-threshold \
            --direction ${direction} \
            --tree_dataset WildChat10K \
            --tree_path stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10] \
            --embedding_dataset ${dataset} \
            --embedding_split full \
            --results_path real/[llama3.2-3b-instruct]BEAT[gemma2-2b-it]
    done
done
