from constants import DQL_DATA_PREFIX
from readsgm import readSGM
from generate import generate_data
import torch


import os


def generate_dql_for(en_fname, translations_per_sentence):
    en_lines = readSGM(en_fname)
    data = generate_data(en_lines, translations_per_sentence)
    os.makedirs(DQL_DATA_PREFIX, exist_ok=True)
    torch.save(data, DQL_DATA_PREFIX+en_fname[:-11]+str(translations_per_sentence)+".pt")

if __name__ == '__main__':
  generate_dql_for("newstest2009-src.en.sgm", 32)
