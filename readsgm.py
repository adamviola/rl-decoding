from bs4 import BeautifulSoup
from constants import DATA_PREFIX

def readSGM(filename):
    with open(DATA_PREFIX + filename, encoding="utf-8") as f:
        file_data = f.read()
    soup = BeautifulSoup(file_data,'html.parser')
    full_text = soup.get_text()
    lines = full_text.split("\n")
    sentances = list(filter(lambda line: len(line) > 0, lines))
    return sentances
