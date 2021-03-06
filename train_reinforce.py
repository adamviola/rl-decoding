import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TestTubeLogger
import torch
from readsgm import readSGM
from reinforce import ReinforceBaseline
from constants import TRAIN_PAIRS, VAL_PAIRS

def load_sentences(paths):
    sentences = []
    for path in paths:
        sentences += readSGM(path)
    return sentences

def train_reinforce(checkpoint_path=None):

    if not torch.cuda.is_available():
        print("\n\nWARNING: No GPU found\n\n")

    batch_size = 32
    epoch_size = 256 # Size of generated training data/epoch
    dataloader_interval = 1 # How many times to re-use generated training data/epoch
    val_interval = 64 # Number of epochs before checking validation

    # Read English sentences from file; replace this later
    train_data = load_sentences([pair[0] for pair in TRAIN_PAIRS])
    val_data = load_sentences([pair[0] for pair in VAL_PAIRS])

    # Create model
    learner = ReinforceBaseline(train_data, val_data, batch_size=batch_size, epoch_size=epoch_size)

    callbacks = [
        ModelCheckpoint(
            monitor='val_mean_log_prob',
            filename='{epoch}-{val_mean_log_prob:.3f}',
            mode='max',
            save_last=True
        ),
        EarlyStopping(monitor='val_mean_log_prob', patience=5, mode='max'),
        TQDMProgressBar()
    ]

    # Train model
    logger = TestTubeLogger("drive/MyDrive/DQL_Data", name="reinforce")
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        reload_dataloaders_every_n_epochs=dataloader_interval,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=val_interval,
        log_every_n_steps=4,
        accumulate_grad_batches=2,
        gpus=1 if torch.cuda.is_available() else 0
    )
    trainer.fit(learner, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    train_reinforce()