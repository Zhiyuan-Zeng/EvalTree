dataset=$1
results_path=$2
split=$3

negative_instance_num=50
positive_instance_num=50
maximum_size=20
seed=0

if [ ! -d "Datasets/${dataset}/eval_results/${results_path}/TextDiff/[negative_instance=${negative_instance_num}]_[positive_instance=${positive_instance_num}]_[maximum=${maximum_size}]_[seed=${seed}]" ]; then
    echo "Run compare.py"
    python -m Baselines.TextDiff.compare --dataset ${dataset} --results_path ${results_path}
else 
    echo "No need to run compare.py"
fi

python -m AssociatedInstances.annotate --dataset ${dataset} --split ${split} --capability_path Datasets/${dataset}/eval_results/${results_path}/TextDiff/[negative_instance=${negative_instance_num}]_[positive_instance=${positive_instance_num}]_[maximum=${maximum_size}]_[seed=${seed}]/weakness-profile.json
python -m Baselines.TextDiff.generate --dataset ${dataset} --results_path ${results_path}  --split ${split}
