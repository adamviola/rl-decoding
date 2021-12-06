import torch
from dqn import DeepQLearning
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def load_data(paths):
    data = []
    for path in paths:
        print(f'Loading data from {path}')
        section = torch.load(path)
        data.extend(section)
    return data


def train_dqn(checkpoint_path=None):
    if not torch.cuda.is_available():
        print("\n\nWARNING: No GPU found\n\n")

    batch_size = 32

    # Read in data
    train_data_paths = [
        'train_data/news-test200832.pt',
        'drive/MyDrive/DQL_Data/newstest200932.pt',
        'drive/MyDrive/DQL_Data/newstest201032.pt',
        'drive/MyDrive/DQL_Data/newstest201132.pt',
        'drive/MyDrive/DQL_Data/newstest201232.pt',
        'drive/MyDrive/DQL_Data/newstest2014-fren32.pt',
    ]
    train_data = load_data(train_data_paths)
    val_data = load_data(['train_data/newsdiscussdev2015-enfr32.pt'])

    # Create dataloaders for data
    train_dataloader = DataLoader(train_data, batch_size, shuffle=True, collate_fn=DeepQLearning.collate_fn)
    val_dataloader = DataLoader(val_data, batch_size, collate_fn=DeepQLearning.collate_fn)

    learner = DeepQLearning()

    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='{epoch}-{val_loss:.3f}'
        ),
        EarlyStopping(monitor='val_loss', patience=5)
    ]

    # Train model
    logger = TestTubeLogger("drive/MyDrive/DQL_Data", name="dqn")
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=2,
        gpus=1 if torch.cuda.is_available() else 0
    )
    trainer.fit(learner, train_dataloader, val_dataloader, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    train_dqn()