import json
import time

import requests

with open("analytics.json", encoding="utf-8") as f:
    data = json.load(f)


def upload(item, i):
    success = False
    while not success:
        try:
            response = requests.get(item.get("utl"))
            with open(f"files/{i}.pdf", "wb") as file:
                file.write(response.content)
            success = True
        except Exception as e:
            time.sleep(3)
            print(f"Error downloading {i}: {e}")


for item in data:
    print(item)
    if "pdf" in item.get("utl"):
        print(F"Doc {item.get("id")}")
        upload(item, item.get("id"))
