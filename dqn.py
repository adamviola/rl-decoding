from transformers import MarianMTModel, MarianTokenizer
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

class DeepQLearning(pl.LightningModule):
    # Constructor; creates Reformer and linear layer
    def __init__(self):
        super().__init__()

        model_name = 'Helsinki-NLP/opus-mt-en-fr'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        # self.linear = nn.Linear(512, 59514)

    # Tells PyTorch Lightning how to do inference
    def forward(self, x):
        values = self.model(*x).logits

        # Equivalent to sigmoid + log
        # Useful because the rewards are always negative (and are often very small)
        return -torch.log(1 + torch.exp(-values))

    # Tells PyTorch Lightning how to do a training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        q = self(x) # approximation of q-values for all actions

        padded_rewards, reward_mask = y
        max_q = q.max(dim=2).values
        max_q[:,:-1] = max_q[:,:-1] * reward_mask
        max_q[:,-1] = 0

        targets = (padded_rewards + max_q[:,1:]).detach() # detach this to avoid backprop

        actions = x[-2][:, 1:] # first "action" was the padding token
        action_values = q[:,:-1].gather(2, actions.unsqueeze(2)).squeeze(2) # Last q-value is for the </s> token

        losses = F.mse_loss(action_values, targets, reduction='none')
        loss = (losses * reward_mask).sum() / reward_mask.sum()

        return loss

    # Tells PyTorch Lightning which optimizer to use
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    # Collate function for PyTorch DataLoader
    def collate_fn(batch):
        tokenized_sentences = [datum[0] for datum in batch]
        tokenized_translations = [datum[1] for datum in batch]
        rewards = [datum[2] for datum in batch]

        encoder_input_ids = nn.utils.rnn.pad_sequence(tokenized_sentences, batch_first=True)
        encoder_mask = nn.utils.rnn.pad_sequence([torch.ones_like(input_id) for input_id in encoder_input_ids], batch_first=True)
        
        decoder_input_ids = nn.utils.rnn.pad_sequence(tokenized_translations, batch_first=True)
        decoder_mask = nn.utils.rnn.pad_sequence([torch.ones_like(input_id) for input_id in decoder_input_ids], batch_first=True)

        padded_rewards = nn.utils.rnn.pad_sequence(rewards, batch_first=True)
        reward_mask = nn.utils.rnn.pad_sequence([torch.ones_like(reward) for reward in rewards], batch_first=True)

        x = (encoder_input_ids, encoder_mask, decoder_input_ids, decoder_mask)
        y = (padded_rewards, reward_mask)

        return (x, y)


# data = [
#     ('how are you', '<pad>', [1]),
#     ('what time is it', '<pad> what time is it', [1,2,3,4,5]),
#     # ('what is the weather', '<pad> lmao', torch.ones(2)),
#     ('hi', '<pad>', [2])
# ]



# model_name = 'Helsinki-NLP/opus-mt-en-fr'
# tokenizer = MarianTokenizer.from_pretrained(model_name)
# model = MarianMTModel.from_pretrained(model_name)

# encoder_input = tokenizer([datum[0] for datum in data], return_tensors='pt', padding=True)
# decoder_input = tokenizer([datum[1] for datum in data], return_tensors='pt', padding=True)

# print(decoder_input)

# encoder_input_ids = encoder_input['input_ids']
# encoder_attention_masks = encoder_input['attention_mask']

# decoder_input_ids = decoder_input['input_ids']
# decoder_attention_masks = decoder_input['attention_mask']

# output = model(encoder_input_ids, encoder_attention_masks, decoder_input_ids, decoder_attention_masks).logits

# print(output.shape)




# blah = 'Hello!'

# model_name = 'Helsinki-NLP/opus-mt-en-fr'

# tokenizer = MarianTokenizer.from_pretrained(model_name)

# enc = tokenizer.encode(blah, return_tensors='pt')
# print(enc)

# dec = tokenizer.decode(enc[0], skip_special_tokens=True)
# print(dec)





# sentences = ["How are you doing on this fine day?"]

# data = generate_data(sentences, 1)


# learner = DeepQLearning()

# data = DataLoader(data, 2, collate_fn=DeepQLearning.collate_fn)

# trainer = pl.Trainer()

# trainer.fit(learner, data)

