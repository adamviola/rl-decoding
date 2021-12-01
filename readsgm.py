from bs4 import BeautifulSoup

def readSGM(filename):
    with open(filename, encoding="utf-8") as f:
        file_data = f.read()
    soup = BeautifulSoup(file_data,'html.parser')
    full_text = soup.get_text()
    lines = full_text.split("\n")
    sentances = list(filter(lambda line: len(line) > 0, lines))
    return sentances
