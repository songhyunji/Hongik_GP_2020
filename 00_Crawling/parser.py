## parser.py
import requests
from bs4 import BeautifulSoup
import json
import os

## python파일의 위치
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'result.json'), 'w+', encoding='UTF-8-sig') as json_file:
    json_file.write("[\n")

req = requests.get('http://www.jobkorea.co.kr/starter/passassay/View/143422')
html = req.text
soup = BeautifulSoup(html, 'html.parser')
my_titles = soup.select(
    'dl > dt > button > span.tx'
    )

data = {}

for title in my_titles:
    data[title.text] = title.get('class="tx"')

res = not data

if not res:
    print('http://www.jobkorea.co.kr/starter/passassay/View/143422')
    with open(os.path.join(BASE_DIR, 'result.json'), 'a+', encoding='UTF-8-sig') as json_file:
        json_file.write(json.dumps(data, ensure_ascii=False, indent=4))

for i in range(143425, 235333):
    req = requests.get('http://www.jobkorea.co.kr/starter/passassay/View/' + str(i))
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    my_titles = soup.select(
        'dl > dt > button > span.tx'
    )

    data = {}

    for title in my_titles:
        data[title.text] = title.get('class="tx"')

    res = not data

    if not res:
        print('http://www.jobkorea.co.kr/starter/passassay/View/' + str(i))
        with open(os.path.join(BASE_DIR, 'result.json'), 'a+', encoding='UTF-8-sig') as json_file:
            json_file.write(",\n")
            json_file.write(json.dumps(data, ensure_ascii=False, indent=4))


with open(os.path.join(BASE_DIR, 'result.json'), 'a+', encoding='UTF-8-sig') as json_file:
    json_file.write("\n]")