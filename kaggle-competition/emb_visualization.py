import os
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np

# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir='logs/imdb-example/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# subwords = ['one', 'two', 'three']
activation = np.load('embeddings.npy')[:5000]

# # Save Labels separately on a line-by-line manner.
# with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
#   for subword in subwords:
#     f.write("{}\n".format(subword))


# Save the weights we want to analyze as a variable. Note that the first
# value represents any unknown word, which is not in the metadata, here
# we will remove this value.
weights = tf.Variable(activation)
# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)