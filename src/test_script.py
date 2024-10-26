import json

# Replace 'file_path.json' with your actual file path
with open('../data/datasets/blender/chair/nb-info.json', 'r') as file:
    data = json.load(file)

print(data)