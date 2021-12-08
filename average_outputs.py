from constants import OUTPUTS_PREFIX
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

METRIC_COLUMNS = ['BLEU Score', '1-Gram', '2-Gram', '3-Gram', '4-Gram']

def average_BLEU_data(filename):
    df = pd.read_csv(OUTPUTS_PREFIX + filename)
    print(filename)
    for cname in METRIC_COLUMNS:
        print(cname + ": " + str(np.mean(df[cname].tolist())))
    # Corpus Bleu score
    refs = [[sentance.split()] for sentance in df["Original French"]]
    hyps = [sentance.split() for sentance in df["Generated French"]]
    corp_bleu = corpus_bleu(refs, hyps)
    print("Corpus Bleu" + ": " + str(corp_bleu))

average_BLEU_data("baseline-greedy.csv")
average_BLEU_data("reinforce-greedy.csv")
average_BLEU_data("baseline-beam.csv")
average_BLEU_data("reinforce-beam.csv")
