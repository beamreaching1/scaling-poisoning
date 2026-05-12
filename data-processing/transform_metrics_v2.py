import csv
from collections import defaultdict

input_file = 'e:/GitHub/scaling-poisoning/data-processing/metrics-13_14_47-30-Apr-26.csv'
output_file = 'e:/GitHub/scaling-poisoning/data-processing/metrics-13_14_47-30-Apr-26.csv' 
# Overwriting original file since the user said "Transform #file... to only contain..."

model_to_epochs = defaultdict(dict)
temp_output_file = 'e:/GitHub/scaling-poisoning/data-processing/metrics-13_14_47-30-Apr-26_temp.csv'

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_name = row.get('model_name') or row.get('run.model_name')
        if not model_name or model_name == '-':
            for k, v in row.items():
                if v and v != '-' and ('model_name' in k):
                    model_name = v
                    break

        if not model_name or model_name == '-':
            continue

        event_prefixes = set()
        for k in row.keys():
            if k and k.startswith('events["'):
                idx = k.rfind('].')
                if idx != -1:
                    prefix = k[:idx + 1]
                    event_prefixes.add(prefix)
                
        for prefix in event_prefixes:
            epoch_key = prefix + '.epoch'
            score_key = prefix + '.metrics.overall_score'
            
            if epoch_key in row and row[epoch_key] and row[epoch_key] != '-' and score_key in row and row[score_key] and row[score_key] != '-':
                try:
                    epoch = int(float(row[epoch_key]))
                    score = float(row[score_key])
                    model_to_epochs[model_name][epoch] = score
                except ValueError:
                    pass

with open(temp_output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow(['model name', 'epoch 0', 'epoch 1', 'epoch 2', 'epoch 3', 'epoch 4', 'epoch 5'])
    for model, epochs in model_to_epochs.items():
        writer.writerow([model] + [epochs.get(e, '') for e in range(6)])

import os
import shutil
os.replace(temp_output_file, input_file)

print('Done!')