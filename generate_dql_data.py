from constants import DQL_DATA_PREFIX
from readsgm import readSGM
from generate import generate_data
import torch
import math



def generate_dql_for(en_fname, translations_per_sentence, batch_size):
    en_lines = readSGM(en_fname)
    for i in range(0, math.ceil(len(en_lines) / batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = en_lines[start:end]
        data = generate_data(batch, translations_per_sentence)
        torch.save(data, DQL_DATA_PREFIX+en_fname[:-10]+str(translations_per_sentence)+"-"+str(i)+".pt")

generate_dql_for("news-test2008-src.en.sgm", 16,50)