# bash build_data.sh
for dataset in MATH WildChat10K Chatbot-Arena Chatbot-Arena_NEW MMLU DS-1000; do
    python -m build_data --dataset ${dataset}
done
