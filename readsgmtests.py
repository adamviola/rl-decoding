from readsgm import readSGM
from constants import TRAIN_PAIRS, DATA_PREFIX, OUTPUTS_PREFIX

for en, fr in TRAIN_PAIRS:
    print(OUTPUTS_PREFIX+en[:-11]+".csv")
    en_loc = DATA_PREFIX+en 
    fr_loc = DATA_PREFIX+fr
    en_count = len(readSGM(en_loc))
    fr_count = len(readSGM(fr_loc))
    print((en_count == fr_count, en_count, fr_count))