from dqn import DeepQLearning
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from generate import generate_data

batch_size = 256

# Some way of reading or generating data; replace this later 
sentences = ["How are you doing on this fine day?"]
train_data = generate_data(sentences, 256)

# Create dataloaders for data
learner = DeepQLearning()
# learner = DeepQLearning.load_from_checkpoint('/path/to/checkpoint')
train_dataloader = DataLoader(train_data, batch_size, collate_fn=DeepQLearning.collate_fn)

# Train model
trainer = pl.Trainer()
trainer.fit(learner, train_dataloader)