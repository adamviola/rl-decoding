
from transformers import MarianMTModel, MarianTokenizer
import torch.nn.functional as F
import torch
import math
from tqdm import tqdm
from constants import VAL_PAIRS
from dqn import DeepQLearning
from readsgm import readSGM
from reinforce import ReinforceBaseline

model_name = 'Helsinki-NLP/opus-mt-en-fr'
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(sentences, num_beams=1, model_class=None, checkpoint_path=None, out='outputs/adam/'):
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    if model_class:
        model = model_class.load_from_checkpoint(checkpoint_path).model.to(device)
    else:
        model = MarianMTModel.from_pretrained(model_name).to(device)

    translations = []

    # Generate translations in batches
    for i in tqdm(range(math.ceil(len(sentences) / batch_size))):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = sentences[start:end]

        # Tokenize batch of sentences
        encoder_input = tokenizer(batch, return_tensors='pt', padding=True)
        input_ids = encoder_input['input_ids'].to(device)
        attention_mask = encoder_input['attention_mask'].to(device)

        # Generate translations
        results = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=num_beams,
            return_dict_in_generate=True
        )

        # Decode translations
        translations.extend(tokenizer.batch_decode(results.sequences, skip_special_tokens=True))

    with open(f"{out}/{model_class.__name__ if model_class else 'pretrained'}-{'greedy' if num_beams == 1 else 'beam'}.txt", 'w') as f:
        for translation in translations:
            f.write(f'{translation}\n')

if __name__ == '__main__':
    with open('data/test_en.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    # predict(sentences)
    # predict(sentences, num_beams=4)
    # predict(sentences, model_class=ReinforceBaseline, checkpoint_path='checkpoints/reinforce-epoch=575-val_mean_log_prob=-5.490.ckpt')
    # predict(sentences, model_class=ReinforceBaseline, checkpoint_path='checkpoints/reinforce-epoch=319-val_mean_log_prob=-6.121.ckpt')
    # predict(sentences, model_class=ReinforceBaseline, checkpoint_path='checkpoints/reinforce-epoch=63-val_mean_log_prob=-8.707.ckpt')
    # predict(sentences, model_class=DeepQLearning, checkpoint_path='checkpoints/dqn-epoch=10-val_loss=0.138.ckpt')
