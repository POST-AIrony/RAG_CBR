# from bs4 import BeautifulSoup, SoupStrainer

# data = []
# soup = BeautifulSoup(
#     open("./register.html", "r", encoding="utf-8").read(), "html.parser"
# )
# c = 1
# docs = soup.find_all("div", class_="cross-result")
# for i in docs:
#     htm = f"""<!DOCTYPE html>
#     <html>
#     <head>
#     <title> Hello </title>
#     </head>
#     <body>
#     {i}
#     </body>
#     </html>
#     """
#     doc_soup = BeautifulSoup(htm, "html.parser")
#     number = doc_soup.find_all("span", class_="number")[0].text
#     date = doc_soup.find_all("span", class_="date")[0].text
#     title = doc_soup.find_all("a")[0].text
#     href = doc_soup.find_all("a")[0]["href"]
#     print(number, date, title, href)
#     data.append({"number": number, "date": date, "title": title, "link": href, "id": c})
#     c += 1

# import json

# with open("register.json", "w", encoding="utf-8") as fp:
#     json.dump(data, fp, ensure_ascii=False)

import json
import time

import requests

with open("register.json", encoding="utf-8") as f:
    data = json.load(f)


def upload(item, i, exc):
    success = False
    while not success:
        try:
            response = requests.get(item.get("link"))
            with open(f"files/{i}.{exc}", "wb") as file:
                file.write(response.content)
            success = True
        except Exception as e:
            time.sleep(3)
            print(f"Error downloading {i}: {e}")


for item in data:
    print(item)
    # if "xls" in item.get("link"):
    #     print(F"Doc {item.get("id")}")
    #     upload(item, item.get("id"), "xls")
    # elif "xlsx" in item.get('link'):
    #     print(F"Doc {item.get("id")}")
    #     upload(item, item.get("id"), "xlsx")
    # else: print(item.get('link').split('.')[-1])
    if "zip" in item.get("link"):
        print(item)
