python -m Baselines.QualEval.stage1-CapabilityDiscovery.initialize --dataset MATH
for round in {1..6}; do
    python -m Baselines.QualEval.stage1-CapabilityDiscovery.shrink --dataset MATH --round $round
done

python -m Baselines.QualEval.stage1-CapabilityDiscovery.initialize --dataset WildChat10K
for round in {1..6}; do
    python -m Baselines.QualEval.stage1-CapabilityDiscovery.shrink --dataset WildChat10K --round $round
done

python -m Baselines.QualEval.stage1-CapabilityDiscovery.initialize --dataset DS-1000
for round in {1..5}; do
    python -m Baselines.QualEval.stage1-CapabilityDiscovery.shrink --dataset DS-1000 --round $round
done
