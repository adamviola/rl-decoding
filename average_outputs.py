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



def compare_BLEU_data(fname1, fname2):
    df1 = pd.read_csv(OUTPUTS_PREFIX + fname1)
    df2 = pd.read_csv(OUTPUTS_PREFIX + fname2)
    bl1 = np.array(df1["BLEU Score"])
    bl2 = np.array(df2["BLEU Score"])
    if bl1.size != bl2.size:
        print("Mismatched sizes " + str(bl1.size) + ", " + str(bl2.size))
        return
    greater = np.zeros_like(bl1)
    greater[bl1 > bl2] = 1
    equal = np.zeros_like(bl1)
    equal[bl1 == bl2] = 1
    lesser = np.zeros_like(bl1)
    lesser[bl1 < bl2] = 1
    print(str(np.sum(greater)) + " scores of " + fname1 + " greater than scores of " + fname2)
    print(str(np.sum(equal)) + " scores of " + fname1 + " equal to scores of " + fname2)
    print(str(np.sum(lesser)) + " scores of " + fname1 + " lesser than scores of " + fname2)



compare_BLEU_data("reinforce-greedy.csv","baseline-greedy.csv")
# average_BLEU_data("baseline-greedy.csv")
# average_BLEU_data("reinforce-greedy.csv")
# average_BLEU_data("baseline-beam.csv")
# average_BLEU_data("reinforce-beam.csv")
