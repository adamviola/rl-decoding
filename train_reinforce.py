import pytorch_lightning as pl
from reinforce import ReinforceBaseline

batch_size = 256

# Read English sentences from file; replace this later
sentences = ["Hello", "How are you?", "What time is it?", "There's a snake in my boot."]

# Create model
learner = ReinforceBaseline(sentences)
# learner = ReinforceBaseline.load_from_checkpoint('/path/to/checkpoint')

# Train model
trainer = pl.Trainer(reload_dataloaders_every_epoch=True)
trainer.fit(learner)