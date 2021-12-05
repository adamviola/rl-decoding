import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from readsgm import readSGM
from reinforce import ReinforceBaseline
from constants import TRAIN_PAIRS, VAL_PAIRS

def load_sentences(paths):
    sentences = []
    for path in paths:
        sentences += readSGM(path)
    return sentences

if not torch.cuda.is_available():
    print("\n\nWARNING: No GPU found\n\n")

batch_size = 32
epoch_size = 256
dataloader_interval = 4
val_interval = 32

checkpoint_path = None

if checkpoint_path:
    # Load checkpoint
    learner = ReinforceBaseline.load_from_checkpoint(checkpoint_path)
else:
    # Read English sentences from file; replace this later
    train_data = load_sentences([pair[0] for pair in TRAIN_PAIRS])
    val_data = load_sentences([pair[0] for pair in VAL_PAIRS])

    # Create model
    learner = ReinforceBaseline(train_data, val_data, batch_size=batch_size, epoch_size=epoch_size)

callbacks = [
    ModelCheckpoint(
        monitor='mean_reward',
        dirpath='./checkpoints/reinforce/',
        filename='{epoch}-{mean_reward:.3f}',
        mode='max',
    ),
    EarlyStopping(monitor='mean_reward', patience=5, mode='max')
]

# Train model
trainer = pl.Trainer(
    callbacks=callbacks,
    reload_dataloaders_every_n_epochs=dataloader_interval,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=val_interval,
    log_every_n_steps=4,
    gpus=1 if torch.cuda.is_available() else 0
)
trainer.fit(learner)