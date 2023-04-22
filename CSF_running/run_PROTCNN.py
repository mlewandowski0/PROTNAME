epochs = 25
experiment_name = "protCNN"

root_data = "."
n_families_of_interest = 3000
data_dirpath = "random_split"
vocab_size = 25
embedding_dim = vocab_size
max_length = 512
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, MultiHeadAttention, Dense, GlobalAveragePooling1D, LayerNormalization, Dropout
import matplotlib.pyplot as plt 
from functools import partial

import os
import pandas as pd
import numpy as np
from collections import Counter

################################################################################
os.makedirs(root_data, exist_ok=True)
os.makedirs(os.path.join(root_data, experiment_name), exist_ok=True)

# Architecture 
def residual_block(x, dil, filters, ks=3):
    shortcut = x
    bn1 = tf.keras.layers.BatchNormalization()(x)
    a1 = tf.keras.layers.Activation("relu")(x)
    conv1 = tf.keras.layers.Conv1D(filters, ks, dilation_rate=dil, padding="same")(x)
    
    bn2 = tf.keras.layers.BatchNormalization()(conv1)
    a2 = tf.keras.layers.Activation("relu")(bn2)
    conv2 = tf.keras.layers.Conv1D(filters, ks, padding="same")(a2)
    
    x = tf.keras.layers.Add()([conv2, shortcut])
    return x

def getProtCNN(numclass, residuals_ks=3, residuals_filters=64):
    input_x = tf.keras.layers.Input(shape=(512, ))
    x = tf.keras.layers.Embedding(vocab_size, vocab_size, embeddings_initializer=tf.keras.initializers.Identity(gain=1.0), trainable=False)(input_x)
    x = tf.keras.layers.Permute(dims=[2, 1])(x)
    x = tf.keras.layers.Conv1D(64, 8, padding="same")(x)
    x = residual_block(x, 1, residuals_filters, ks=residuals_ks)
    x = residual_block(x, 2, residuals_filters, ks=residuals_ks)
    x = tf.keras.layers.Permute(dims=[2, 1])(x)
    x = tf.keras.layers.Conv1D(64, 3, padding="same")(x)
    x = residual_block(x, 1, residuals_filters, ks=residuals_ks)
    x = residual_block(x, 2, residuals_filters, ks=residuals_ks)
    x = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, -1))(x)
    x = tf.keras.layers.Conv2D(32, (4,4), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(8, (8,8), padding="same")(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((64,64))(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(numclass+1, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_x, outputs=out)
    return model

################################################################################
# FUNCTION & CLASSES

class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A LearningRateSchedule that uses cyclical schedule."""
    def __init__(
        self,
        initial_learning_rate,
        maximal_learning_rate,
        step_size,
        scale_fn,
        scale_mode: str = "cycle",
        name: str = "CyclicalLearningRate",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.step_size = step_size
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "CyclicalLearningRate"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
            step_size = tf.cast(self.step_size, dtype)
            step_as_dtype = tf.cast(step, dtype)
            cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
            x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)

            mode_step = cycle if self.scale_mode == "cycle" else step

            return initial_learning_rate + (
                maximal_learning_rate - initial_learning_rate
            ) * tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(mode_step)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "maximal_learning_rate": self.maximal_learning_rate,
            "scale_fn": self.scale_fn,
            "step_size": self.step_size,
            "scale_mode": self.scale_mode,
        }

def read_all_shards(partition='dev', data_dir = data_dirpath):
    shards = []
    for fn in os.listdir(os.path.join(data_dir, partition)):
        with open(os.path.join(data_dir, partition, fn)) as f:
            shards.append(pd.read_csv(f, index_col=None))
    
    return pd.concat(shards)

def read_all_data_initial():
  global train, test, dev, all_train_ds_size, all_test_ds_size, all_dev_ds_size

  test = read_all_shards('test')
  dev = read_all_shards('dev')
  train = read_all_shards('train')

  partitions = {'test': test, 'dev': dev, 'train': train}
  for name, df in partitions.items():
      print('Dataset partition "%s" has %d sequences' % (name, len(df)))

  all_train_ds_size = len(train)
  all_test_ds_size = len(test)
  all_dev_ds_size = len(dev)

  train.reset_index(inplace=True, drop=True)
  dev.reset_index(inplace=True, drop=True)
  test.reset_index(inplace=True, drop=True)

def get_cumulative(data):
    counter = Counter(data['family_accession'])
    print(f"how many labels : {len(counter)}")
    
    datasetSize = len(data)
    xs = []
    x_labels = []
    ys = []

    t = 0
    cumulative = []

    for i,(x, y) in  enumerate(counter.most_common()):
        xs.append(i)
        x_labels.append(x)
        ys.append(y)
        t += y / datasetSize
        cumulative.append(t)
    return cumulative

def plot_history(data):
  plt.plot(data['loss'], label="loss")
  plt.plot(data['val_loss'], label="val_loss")
  plt.legend()
  plt.savefig(os.path.join(root_data, experiment_name, f"loss_plot.png"))

  plt.plot(data['accuracy'], label="accuracy")
  plt.plot(data['val_accuracy'], label="val_accuracy")
  plt.legend()
  plt.savefig(os.path.join(root_data, experiment_name, f"accuracy_plot.png"))

def train_network(get_model, save_path, epochs=25, INIT_LR=1e-4, MAX_LR=1e-3, BATCH_SIZE=32):
  steps_per_epoch = len(train_sequences) // BATCH_SIZE
  clr = CyclicalLearningRate(initial_learning_rate=INIT_LR,
      maximal_learning_rate=MAX_LR,
      scale_fn=lambda x: 1/(2.**(x-1)),
      step_size=2 * steps_per_epoch
  )

  es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', min_delta=0.01, patience=5)
  save_model_cb = tf.keras.callbacks.ModelCheckpoint(save_path, monitor = "accuracy", verbose= 1, save_best_only = True) 

  model = get_model()
  optimizer = tf.keras.optimizers.Adam(clr)
  model.compile(loss='sparse_categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
  model.summary()

  history = model.fit(train_padded, training_label_seq, epochs=epochs, validation_data=(validation_padded, validation_label_seq), 
                      callbacks = [es_cb , save_model_cb])
  
  data = history.history
  plot_history(data)
  return history

################################################################################
# EXECUTION CODE
print('Available dataset partitions: ', os.listdir(data_dirpath))
read_all_data_initial()
cumulative = get_cumulative(train)
print(f"{n_families_of_interest} classes is {100 * round( cumulative[n_families_of_interest-1],3)} portion of training data")

################################################################################
familiesOfInterest = train.family_accession.value_counts()[:n_families_of_interest]

mask = train.family_accession.isin(familiesOfInterest.index.values)
train = train.loc[mask,:]

mask = dev.family_accession.isin(familiesOfInterest.index.values)
dev = dev.loc[mask,:]

mask = test.family_accession.isin(familiesOfInterest.index.values)
test = test.loc[mask,:]


################################################################################
train_seq = train['sequence']
dev_seq = dev['sequence']
test_seq = test['sequence']

################################################################################
train_sentences = train_seq.apply(lambda seq: [aa for aa in seq])
validation_sentences = dev_seq.apply(lambda seq: [aa for aa in seq])
test_sentences = test_seq.apply(lambda seq: [aa for aa in seq])

################################################################################
train_labels = train['family_accession'].apply(lambda x: x.split('.')[0])
validation_labels = dev['family_accession'].apply(lambda x: x.split('.')[0])
test_labels = test['family_accession'].apply(lambda x: x.split('.')[0])

label_tokenizer = Tokenizer(oov_token = -1)
label_tokenizer.fit_on_texts(train_labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))
test_label_seq = np.array(label_tokenizer.texts_to_sequences(test_labels))

training_label_seq = training_label_seq-1
validation_label_seq = validation_label_seq-1
test_label_seq = test_label_seq-1

################################################################################
tokenizer = Tokenizer(oov_token=oov_tok, num_words = vocab_size)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

################################################################################
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type)

validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type)

################################################################################
model = partial(getProtCNN, n_families_of_interest)
train_network(model, os.path.join(root_data, experiment_name, "model.h5"), epochs=epochs)