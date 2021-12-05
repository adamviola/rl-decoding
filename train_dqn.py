import torch
from dqn import DeepQLearning
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from generate import generate_data

def load_data(paths):
    data = []
    for path in paths:
        section = torch.load(path)
        data.extend(section)
    return data

if not torch.cuda.is_available():
    print("\n\nWARNING: No GPU found\n\n")

batch_size = 32

checkpoint_path = None

# Read in data
train_data_paths = [
    'train_data/news-test200832.pt',
    'train_data/newstest200932.pt',
    'train_data/newstest201032.pt',
    'train_data/newstest201132.pt',
    'train_data/newstest201232.pt',
    'train_data/newstest2014-fren32.pt',
] # Missing some I think
train_data = load_data(train_data_paths)
# val_data = load_data(['path/to/validation/data']) # Doesn't exist yet

# Create dataloaders for data
train_dataloader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=DeepQLearning.collate_fn)
# val_dataloader = DataLoader(val_data, batch_size, shuffle=True, collate_fn=DeepQLearning.collate_fn)

if checkpoint_path:
    model = DeepQLearning.load_from_checkpoint(checkpoint_path)
else:
    learner = DeepQLearning()

callbacks = [
    ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/dqn/',
        filename='{epoch}-{val_loss:.3f}'
    ),
    EarlyStopping(monitor='val_loss', patience=5)
]

# TODO: Implement validation for DQN, our own logging, fixed baseline based on greedy?, trainer.validate(model) before training

# Train model
trainer = pl.Trainer(callbacks=callbacks)
trainer.fit(learner, train_dataloader) #, val_dataloader)