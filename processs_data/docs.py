from bs4 import BeautifulSoup, SoupStrainer

data = []
"""<div class="cross-result" data-doc-id="7689">
<div class="date-number">
<span class="number col-md-3">№ ИН-018-34/17</span>
<span class="date col-md-2">от 29.02.2024</span>
</div>
<div class="title-source offset-md-4">
<div class="title">
<a data-zoom-tags="" data-zoom-title="Информационное письмо №ИН-018-34/17 от 29.02.2024 «Информационное письмо Банка России об отдельных видах финансовых инструментов и цифровых финансовых активов»" href="https://cbr.ru/Crosscut/LawActs/File/7689" target="_blank">Информационное письмо Банка России об отдельных видах финансовых инструментов и цифровых финансовых активов</a>
</div>
<div class="source">Информационное письмо</div>
</div>
</div>"""
soup = BeautifulSoup(open("./page.html", "r", encoding="utf-8").read(), "html.parser")
docs = soup.find_all("div", class_="cross-result")
for i in docs:
    htm = f"""<!DOCTYPE html>
    <html>
    <head>
    <title> Hello </title>
    </head>
    <body>
    {i}
    </body>
    </html>
    """
    doc_soup = BeautifulSoup(htm, "html.parser")
    number = doc_soup.find_all("span", class_="number")[0].text
    date = doc_soup.find_all("span", class_="date")[0].text
    title = doc_soup.find_all("a")[0].text
    href = doc_soup.find_all("a")[0]["href"]
    data.append({"number": number, "date": date, "title": title, "link": href})
print(data)
