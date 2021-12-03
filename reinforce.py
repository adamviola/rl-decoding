from transformers import MarianMTModel, MarianTokenizer
import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generate import generate_data

class ReinforceBaseline(pl.LightningModule):
    def __init__(self, sentences, batch_size=128, epoch_size=2048):
        super().__init__()
        model_name = 'Helsinki-NLP/opus-mt-en-fr'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        # self.v_linear = nn.Linear(512, 1)

        self.sentences = sentences
        self.batch_size = batch_size
        self.epoch_size = epoch_size

    # Tells PyTorch Lightning how to do inference
    def forward(self, x):
        return F.log_softmax(self.model(*x).logits, dim=2)


        # old value stuff
        # output = self.model.model(*x, output_hidden_states=True)['last_hidden_state']
        # action_log_probs = F.log_softmax(self.model.lm_head(output) + self.model.final_logits_bias, dim=2)
        # return self.model(*x)

    # Tells PyTorch Lightning how to do a training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        action_log_probs = self(x)

        padded_rewards, reward_mask = y
        batch_size, seq_len = padded_rewards.shape

        # Compute returns starting from each state (including 0 return at </s> for value function)
        returns = torch.zeros((batch_size, seq_len))
        returns[:, -1] = padded_rewards[:, -1]
        for i in range(seq_len - 2, -1, -1):
            returns[:, i] = padded_rewards[:, i] + returns[:, i + 1] * reward_mask[:, i] # reward mask not necessary?

        actions = x[-2][:, 1:] # First "action" was the padding token

        # Compute baseline: mean return across batch computed independently for each timestep
        baseline = torch.sum(returns * reward_mask, dim=0, keepdim=True) / torch.sum(reward_mask, dim=0, keepdim=True)

        advantages = returns - baseline

        # Last advantage and action_log_prob is for the </s> token, where we don't really take an action
        policy_losses = -advantages * action_log_probs[:,:-1].gather(2, actions.unsqueeze(2)).squeeze(2) * reward_mask

        return policy_losses.sum() / reward_mask.sum()


        # Baseline value function
        # x, y = batch
        
        # action_log_probs, values = self(x)

        # padded_rewards, reward_mask = y
        # batch_size, seq_len = padded_rewards.shape

        # # Compute returns starting from each state (including 0 return at </s> for value function)
        # returns = torch.zeros((batch_size, seq_len + 1))
        # returns[:, -2] = padded_rewards[:, -1]
        # for i in range(seq_len - 2, -1, -1):
        #     returns[:, i] = padded_rewards[:, i] + returns[:, i + 1] * reward_mask[:, i] # reward mask not necessary?

        # actions = x[-2][:, 1:] # First "action" was the padding token
        # value_mask = x[-1]

        # advantages = returns - values.detach()

        # value_losses = F.mse_loss(values, returns, reduction='none') * value_mask
        # policy_losses = -advantages[:,:-1] * action_log_probs[:,:-1].gather(2, actions.unsqueeze(2)).squeeze(2) * reward_mask
        # # Last advantage and action_log_prob is for the </s> token, where we don't really take an action

        # value_loss = value_losses.sum() / value_mask.sum()
        # policy_loss = policy_losses.sum() / reward_mask.sum()

        # return value_loss + policy_loss

    # Tells PyTorch Lightning which optimizer to use
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def train_dataloader(self):
        print(f"Generating {self.epoch_size} new translations and rewards...")
        data = generate_data(list(np.random.choice(self.sentences, size=self.epoch_size)), 1, model=self.model)
        return DataLoader(data, self.batch_size, collate_fn=ReinforceBaseline.collate_fn)

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
#     ('how are you', '<pad>', torch.ones(1)),
#     ('what time is it', '<pad> what time is it', torch.ones(5)),
#     ('hi', '<pad>', torch.ones(1))
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


# sentences = ["Hello!", "What time is it?", "How are you?"]
# data = generate_data(sentences, 1)

# learner = ReinforceBaseline()

# data = DataLoader(data, 3, collate_fn=ReinforceBaseline.collate_fn)

# trainer = pl.Trainer()

# trainer.fit(learner, data)
