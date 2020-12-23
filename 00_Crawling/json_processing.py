import json # import json module

# with statement
with open('result.json', 'rt', encoding='UTF-8-sig') as json_file:
    json_data = json.load(json_file)

print(json_data[0])
print(type(json_data[0]))
questions = []
for i in range(len(json_data)):
    for key in json_data[i].keys():
        questions.append(key)

print(questions)