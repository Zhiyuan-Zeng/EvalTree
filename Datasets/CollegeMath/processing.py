'''bash
wget https://raw.githubusercontent.com/microsoft/unilm/refs/heads/master/mathscale/MWPBench/data/full_test.json
mv full_test.json full_test.jsonl
python processing.py
rm -r full_test.jsonl
'''

import json
with open("full_test.jsonl", "r") as fin :
    dataset = [json.loads(line) for line in fin]
    dataset = [instance for instance in dataset if instance["data_topic"].startswith("college_math")]
with open("CollegeMath.json", "w") as fout :
    json.dump(dataset, fout, indent = 2)