import csv
from collections import defaultdict
import glob

input_file = 'e:/GitHub/scaling-poisoning/data-processing/metrics-13_14_47-30-Apr-26.csv'
output_file = 'e:/GitHub/scaling-poisoning/data-processing/metrics-13_14_47-30-Apr-26.csv' 

model_to_epochs = defaultdict(dict)

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        model_name = row.get('run.model_name')
        if not model_name or model_name == '-':
            model_name = row.get('model_name')
            
        if not model_name or model_name == '-':
            continue

        # Look for events that match epoch and score
        # e.g., events["..."].epoch and events["..."].metrics.overall_score
        
        event_prefixes = set()
        for k in row.keys():
            if k is not None and k.startswith('events["') and '].' in k:
                prefix = k.rsplit('].', 1)[0] + ']'
                event_prefixes.add(prefix)
                
        for prefix in event_prefixes:
            epoch_key = prefix + '.epoch'
            score_key = prefix + '.metrics.overall_score'
            
            if epoch_key in row and row[epoch_key] not in (None, '', '-'):
                if score_key in row and row[score_key] not in (None, '', '-'):
                    try:
                        epo = float(row[epoch_key])
                        # round down to int to get 0-5
                        epo = int(epo)
                        score = float(row[score_key])
                        model_to_epochs[model_name][epo] = score
                    except Exception as e:
                        print("Error parsing", e)

# Generate output rows
output_rows = []
for model, epochs in model_to_epochs.items():
    r = [model]
    for e in range(6):
        r.append(epochs.get(e, ''))
    output_rows.append(r)

with open(output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    writer.writerow(['model name', 'epoch 0', 'epoch 1', 'epoch 2', 'epoch 3', 'epoch 4', 'epoch 5'])
    for r in output_rows:
        writer.writerow(r)

print("Done. Output written to", output_file)
