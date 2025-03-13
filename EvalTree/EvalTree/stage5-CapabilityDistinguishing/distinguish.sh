# bash EvalTree/stage5-CapabilityDistinguishing/distinguish.sh
for dataset in MATH WildChat10K Chatbot-Arena Chatbot-Arena_NEW MMLU DS-1000; do
    python -m EvalTree.stage5-CapabilityDistinguishing.distinguish \
        --dataset ${dataset}
done
