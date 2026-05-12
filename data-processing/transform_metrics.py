import csv
from collections import defaultdict

input_file = 'e:/GitHub/scaling-poisoning/data-processing/metrics-13_14_47-30-Apr-26.csv'
output_file = 'e:/GitHub/scaling-poisoning/data-processing/transformed_metrics.csv'

model_to_epochs = defaultdict(dict)

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        event_prefixes = set()
        for k in row.keys():
            if k and k.startswith('events["'):
                idx = k.rfind('].')
                if idx != -1:
                    prefix = k[:idx + 1]
                    event_prefixes.add(prefix)
                
        for prefix in event_prefixes:
            model_key = prefix + '.run.model_name'
            epoch_key = prefix + '.epoch'
            score_key = prefix + '.metrics.overall_score'
            
            if model_key in row and row[model_key] and epoch_key in row and row[epoch_key] and score_key in row and row[score_key]:
                model_name = row[model_key]
                try:
                    epoch = int(float(row[epoch_key]))
                    score = float(row[score_key])
                    model_to_epochs[model_name][epoch] = score
                except ValueError:
                    pass

with open(output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow(['model name', 'epoch 0', 'epoch 1', 'epoch 2', 'epoch 3', 'epoch 4', 'epoch 5'])
    for model, epochs in model_to_epochs.items():
        writer.writerow([model] + [epochs.get(e, '') for e in range(6)])

print('Done!')
