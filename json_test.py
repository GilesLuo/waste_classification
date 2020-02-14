import json

with open('./train.json', 'rb') as f:
    y_dict = json.loads(f.read())
    # print(y_dict['annotations'])
for i in range(len(y_dict['annotations'])):
    print(y_dict['annotations'][i])
