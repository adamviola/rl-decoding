from transformers import MarianMTModel, MarianTokenizer
import torch.nn.functional as F
import torch
import math
from tqdm import tqdm

model_name = 'Helsinki-NLP/opus-mt-en-fr'
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Sample translations from English sentences using the given model
# Returns the rewards (log-probs) for each sentence using pre-trained MarianMTModel
def generate_data(sentences, translations_per_sentence, model=None):
    if device == 'cpu':
        print('Warning: Running on CPU')

    tokenizer = MarianTokenizer.from_pretrained(model_name)
    evaluator = MarianMTModel.from_pretrained(model_name).to(device)

    if model is None:
        model = evaluator
    model.to("cuda")

    pad_token_id = tokenizer.pad_token_id

    sentences = sentences * translations_per_sentence

    tokenized_sentences = []
    tokenized_translations = []
    rewards = []

    # Generate translations in batches
    for i in tqdm(range(0, math.ceil(len(sentences) / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = sentences[start:end]

        # Tokenize batch of sentences
        encoder_input = tokenizer(batch, return_tensors='pt', padding=True).to("cuda")
        input_ids = encoder_input['input_ids'].to("cuda")
        attention_mask = encoder_input['attention_mask'].to("cuda")

        input_lengths = attention_mask.sum(dim=1).to("cuda")
        for input_id, input_length in zip(input_ids, input_lengths):
            tokenized_sentences.append(input_id[:input_length].cpu())

        # Generate translations
        results = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=256,
            num_beams=1,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True
        )

        # We pass tensors around instead of strings; noticed differences between tensor and decoded-encoded tensor
        # # Decode translations
        # translations.extend(tokenizer.batch_decode(results.sequences, skip_special_tokens=True))

        # Compute unpadded tokenized translations
        decoder_input_ids = results.sequences
        decoder_attention_mask = torch.zeros_like(decoder_input_ids).to(device)
        for i, sequence in enumerate(decoder_input_ids):
            pad_indices = (sequence[1:] == pad_token_id).nonzero()
            seq_len = len(sequence) - 1 if len(pad_indices) == 0 else pad_indices[0].item()

            tokenized_translations.append(sequence[:seq_len + 1].cpu())
            decoder_attention_mask[i, :seq_len] = 1

        # Compute reward using evaluator model (log probs)
        if evaluator != model:
            logits = torch.stack(results.scores, dim=1)
        else:
            with torch.no_grad():
                logits = evaluator(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask).logits
        
        log_probs = F.log_softmax(logits, dim=2)
        for i, sequence in enumerate(decoder_input_ids):
            seq_len = torch.sum(decoder_attention_mask[i])
            seq_log_probs = log_probs[i, :seq_len]
            seq_rewards = seq_log_probs[range(seq_len), sequence[1:seq_len + 1]]
            rewards.append(seq_rewards.cpu())

    return list(zip(tokenized_sentences, tokenized_translations, rewards))


if __name__ == '__main__':
    sentences = ['Hello!', 'How are you?',  'I am fine.', 'I am fine.', 'Thank you Thank you Thank you Thank you Thank you Thank you Thank you']
    data = generate_data(sentences, 1)
    print(data[0])