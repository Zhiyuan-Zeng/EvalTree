# MATH
python -m Assessments.Extrinsic.data.generate_input \
    --dataset MATH \
    --capability_path Assessments/Extrinsic/data/generic-capability/MATH.json \
    --data_size 5000
python -m Assessments.Extrinsic.data.generate_input \
    --dataset MATH \
    --capability_path Datasets/MATH/eval_results/real/Llama-3.1-8B-Instruct/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/[split=4k-1k]weakness-profiles_[size=9]_[with-instances].json
python -m Assessments.Extrinsic.data.generate_input \
    --dataset MATH \
    --capability_path Datasets/MATH/eval_results/real/Llama-3.1-8B-Instruct/QualEval/[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=5]_[direction=lower]_[split=4k-1k]/weakness-profiles_[size=9]_[with-instances].json
python -m Assessments.Extrinsic.data.generate_input \
    --dataset MATH \
    --capability_path Datasets/MATH/eval_results/real/Llama-3.1-8B-Instruct/EvalTree/TREE=[stage3-RecursiveClustering]_[split=4k-1k]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[threshold=0.4]_[with-instances].json





# DS-1000
python -m Assessments.Extrinsic.data.generate_input \
    --dataset DS-1000 \
    --capability_path Assessments/Extrinsic/data/generic-capability/DS-1000.json \
    --data_size 1024
python -m Assessments.Extrinsic.data.generate_input \
    --dataset DS-1000 \
    --capability_path Datasets/DS-1000/eval_results/real/deepseek-coder-6.7b-base/TextDiff/[negative_instance=50]_[positive_instance=50]_[maximum=20]_[seed=0]/[split=600-400]weakness-profiles_[size=5]_[with-instances].json
python -m Assessments.Extrinsic.data.generate_input \
    --dataset DS-1000 \
    --capability_path Datasets/DS-1000/eval_results/real/deepseek-coder-6.7b-base/QualEval/[chunk=20]_[model=gpt-4o-mini]_[num=20]_[factor=4]_[round=4]_[direction=lower]_[split=600-400]/weakness-profiles_[size=5]_[with-instances].json
python -m Assessments.Extrinsic.data.generate_input \
    --dataset DS-1000 \
    --capability_path Datasets/DS-1000/eval_results/real/deepseek-coder-6.7b-base/EvalTree/TREE=[stage3-RecursiveClustering]_[split=600-400]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]/weakness-profiles/[description=gpt-4o-mini]_[direction=lower]_[alpha=0.05]_[threshold=0.4]_[with-instances].json
