import json

with open('preprocess/anet_actions.json', mode='r') as f:
    annos = json.load(f)

nodes = [node['nodeName'] for node in annos['taxonomy']]

with open('data/anet/actions.json', mode='w') as f:
    json.dump(nodes, f)