# bash EvalTree/stage4-CapabilityDescription/describe.sh
for dataset in MATH WildChat10K Chatbot-Arena Chatbot-Arena_NEW MMLU DS-1000; do
    python -m EvalTree.stage4-CapabilityDescription.describe \
        --dataset ${dataset}
done
