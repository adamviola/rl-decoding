from transformers import MarianTokenizer, MarianMTModel
from constants import TEST_PAIRS, DATA_PREFIX, OUTPUTS_PREFIX
from tqdm import tqdm
from readsgm import readSGM
from nltk.translate.bleu_score import sentence_bleu
import pandas as pd


def run_base_case(is_greedy = False):
    beam_count = 20
    if is_greedy:
        beam_count = 1
    ### Import best model
    # Filler Model Replace Later
    model_name = 'Helsinki-NLP/opus-mt-en-fr'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    output_data = {}
    for en_fname, fr_fname in TEST_PAIRS:
        en_loc = DATA_PREFIX+en_fname
        fr_loc = DATA_PREFIX+fr_fname
        en_lines = readSGM(en_loc)
        fr_lines = readSGM(fr_loc)
        output_data["Original English"] = []
        output_data["Original French"] = []
        output_data["Generated French"] = []
        output_data["BLEU Score"] = []
        output_data["1-Gram"] = []
        output_data["2-Gram"] = []
        output_data["3-Gram"] = []
        output_data["4-Gram"] = []
        for en_line, fr_line in tqdm(list(zip(en_lines, fr_lines))):
            inputs = tokenizer.encode(en_line, return_tensors="pt")
            generation_output = model.generate(input_ids=inputs, num_beams=beam_count, return_dict_in_generate=True, output_scores=True)
            output = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
            output_data["Original English"].append(en_line)
            output_data["Original French"].append(fr_line)
            output_data["Generated French"].append(output)
            output_data["BLEU Score"].append(sentence_bleu([fr_line.split()], output.split()))
            output_data["1-Gram"].append(sentence_bleu([fr_line.split()], output.split(), weights=(1, 0, 0, 0)))
            output_data["2-Gram"].append(sentence_bleu([fr_line.split()], output.split(), weights=(0, 1, 0, 0)))
            output_data["3-Gram"].append(sentence_bleu([fr_line.split()], output.split(), weights=(0, 0, 1, 0)))
            output_data["4-Gram"].append(sentence_bleu([fr_line.split()], output.split(), weights=(0, 0, 0, 1)))
        # EndFor
        df=pd.DataFrame.from_dict(output_data,orient='index').transpose()
        if is_greedy:
            df.to_csv(OUTPUTS_PREFIX+en_fname[:-11]+"-greedy.csv")
        else:
            df.to_csv(OUTPUTS_PREFIX+en_fname[:-11]+"-beam.csv")

run_base_case()