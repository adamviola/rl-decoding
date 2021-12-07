from constants import OUTPUTS_PREFIX
import pandas as pd
import numpy as np

METRIC_COLUMNS = ['BLEU Score', '1-Gram', '2-Gram', '3-Gram', '4-Gram']

def average_BLEU_data(filename):
    df = pd.read_csv(OUTPUTS_PREFIX + filename)
    print(filename)
    for cname in METRIC_COLUMNS:
        print(cname + ": " + str(np.mean(df[cname].tolist())))

average_BLEU_data("baseline-beam.csv")
average_BLEU_data("reinforce-beam.csv")