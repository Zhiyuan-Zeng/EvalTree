for dataset in MATH DS-1000; do
    python -m Assessments.Extrinsic.data.generate_output --dataset ${dataset} --source [input-generation=gpt-4o-mini]
    python -m Assessments.Extrinsic.data.generate_output --dataset ${dataset} --source original
done
