import json
with open('data/train.json') as f:
    data = json.load(f)
total = 0
o_count = 0
for ex in data['examples']:
    for slot in ex['slots']:
        total += 1
        if slot == 'O':
            o_count += 1
print(f'Total slots: {total}')
print(f'O labels: {o_count} ({100.0*o_count/total:.2f}%)')
print(f'Entity labels: {total-o_count} ({100.0*(total-o_count)/total:.2f}%)')