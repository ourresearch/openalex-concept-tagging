import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import json
import math
import argparse
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from typeguard import typechecked
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from datetime import datetime


class CustomModel(tf.keras.Model):
    def train_step(self, inputs):
        old_features, labels = inputs
        labels = tf.RaggedTensor.from_tensor(labels, padding=0)
        paper_titles = old_features[0].to_tensor(shape=[None, 32])
        abstracts = old_features[1].to_tensor(shape=[None, 256])
        
        features = (paper_titles, abstracts, old_features[-2], old_features[-1])
        labels = encoding_layer(labels)

        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            loss = loss_fn(labels, predictions)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        metric_1.update_state(labels, predictions)
        metric_2.update_state(labels, predictions)
        metric_3.update_state(labels, predictions)
        metric_4.update_state(labels, predictions)
        metric_5.update_state(labels, predictions)
        metric_6.update_state(labels, predictions)

        return {"loss": tf.reduce_mean(loss, axis=-1), 
                "accuracy": metric_1.result(), 
                "recall": metric_2.result(), 
                "precision": metric_3.result(), 
                "topK15": metric_4.result(),
                "fbeta": metric_5.result(),
                "f1": metric_6.result()}
  
    def test_step(self, inputs):
        old_features, labels = inputs
        labels = tf.RaggedTensor.from_tensor(labels, padding=0)
        paper_titles = old_features[0].to_tensor(shape=[None, 32])
        abstracts = old_features[1].to_tensor(shape=[None, 256])
        
        features = (paper_titles, abstracts, old_features[-2], old_features[-1])
        labels = encoding_layer(labels)

        with tf.GradientTape() as tape:
            predictions = self(features, training=False)
            loss = loss_fn(labels, predictions)

        metric_1.update_state(labels, predictions)
        metric_2.update_state(labels, predictions)
        metric_3.update_state(labels, predictions)
        metric_4.update_state(labels, predictions)
        metric_5.update_state(labels, predictions)
        metric_6.update_state(labels, predictions)

        return {"loss": tf.reduce_mean(loss, axis=-1), 
                "accuracy": metric_1.result(), 
                "recall": metric_2.result(), 
                "precision": metric_3.result(), 
                "topK15": metric_4.result(),
                "fbeta": metric_5.result(),
                "f1": metric_6.result()}
    
    @property
    def metrics(self):
        return [metric_1, metric_2, metric_3, metric_4, metric_5, metric_6]


def _parse_function(example_proto):

    feature_description = {
        'paper_title': tf.io.RaggedFeature(tf.int64),
        'abstract': tf.io.RaggedFeature(tf.int64),
        'journal': tf.io.FixedLenFeature((1,), tf.int64),
        'doc_type': tf.io.FixedLenFeature((1,), tf.int64),
        'targets': tf.io.FixedLenFeature((15,), tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    paper_title = example['paper_title']
    abstract = example['abstract']
    doc_type = example['doc_type']
    journal = example['journal']
    targets = example['targets']

    return (paper_title, abstract, doc_type, journal), targets


def attention_module(query, key, value, i, num_heads, emb_dim=512, name='title'):
    # Multi headed self-attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=emb_dim // num_heads,
        name=f"{name}_encoder_{i}/multiheadattention",
    )(query, key, value)
    attention_output = tf.keras.layers.Dropout(0.1, name=f"{name}_encoder_{i}/att_dropout")(
        attention_output
    )
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"{name}_encoder_{i}/att_layernormalization"
    )(query + attention_output)

    # Feed-forward layer
    ffn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(emb_dim),
        ],
        name=f"{name}_encoder_{i}/ffn",
    )
    ffn_output = ffn(attention_output)
    ffn_output = tf.keras.layers.Dropout(0.1, name=f"{name}_encoder_{i}/ffn_dropout")(
        ffn_output
    )
    sequence_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name=f"{name}_encoder_{i}/ffn_layernormalization"
    )(attention_output + ffn_output)
    return sequence_output


def get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0
            else np.zeros(d_emb)
            for pos in range(max_len)
        ]
    )
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc


def get_dataset(path, data_type='train'):
    
    tfrecords = [f"{path}{data_type}/{x}" for x in os.listdir(f"{path}{data_type}/") if x.endswith('tfrecord')]
    tfrecords.sort()
    
    
    raw_dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO)

    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTO)

    parsed_dataset = parsed_dataset.apply(tf.data.experimental.dense_to_ragged_batch(1024,drop_remainder=True))
    return parsed_dataset


def scheduler(epoch, curr_lr):
    rampup_epochs = 20
    exp_decay = 0.17
    def lr(epoch, beg_lr, rampup_epochs, exp_decay):
        if epoch < rampup_epochs:
            return beg_lr
        else:
            return beg_lr * math.exp(-exp_decay * epoch)
    return lr(epoch, start_lr, rampup_epochs, exp_decay)


parser = argparse.ArgumentParser(description='parsing arguments to set config')

parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=20)
parser.add_argument('-emb', '--embSize', action='store', dest='emb_size', default=64)
parser.add_argument('-lr', '--learningRate', action='store', dest='learning_rate', default='0.0001')
parser.add_argument('-g', '--gamma', action='store', dest='gamma', default='2.0')
parser.add_argument('-b', '--beta', action='store', dest='beta', default='0.90')
parser.add_argument('-lm', '--loadModel', action='store', dest='load_model', default='no')
parser.add_argument('-sm', '--savedModel', action='store', dest='saved_model', default='none')
parser.add_argument('-nl', '--numLayers', action='store', dest='num_layers', default='1')
parser.add_argument('-nh', '--numHeads', action='store', dest='num_heads', default='4')
parser.add_argument('-d1', '--dense1', action='store', dest='dense_1', default='2048')
parser.add_argument('-d2', '--dense2', action='store', dest='dense_2', default='1024')
parser.add_argument('-d3', '--dense3', action='store', dest='dense_3', default='1024')


args = parser.parse_args()


mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    AUTO = tf.data.experimental.AUTOTUNE

    model_iteration = 'iteration_2'
    global start_lr
    start_lr = float(args.learning_rate)

    with open(f"./data/{model_iteration}/vocab/topics_vocab.pkl", "rb") as f:
        target_vocab = pickle.load(f)

    with open(f"./data/{model_iteration}/vocab/doc_type_vocab.pkl", "rb") as f:
        doc_vocab = pickle.load(f)

    with open(f"./data/{model_iteration}/vocab/journal_name_vocab.pkl", "rb") as f:
        journal_vocab = pickle.load(f)
        
    with open(f"./data/{model_iteration}/vocab/paper_title_vocab.pkl", "rb") as f:
        title_vocab = pickle.load(f)

    print(f"Topic vocab len: {len(target_vocab)}")


    encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=len(target_vocab)+1, output_mode="binary", sparse=False)
    

    loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=float(args.beta), gamma=float(args.gamma), 
                                       reduction=tf.keras.losses.Reduction.NONE)

    metric_1 = tf.keras.metrics.CategoricalAccuracy()
    metric_2 = tf.keras.metrics.Recall(thresholds=0.50)
    metric_3 = tf.keras.metrics.Precision(thresholds=0.50)
    metric_4 = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
    metric_5 = tfa.metrics.FBetaScore(num_classes=len(target_vocab)+1, beta=0.5, threshold=0.50, average='micro')
    metric_6 = tfa.metrics.F1Score(num_classes=len(target_vocab)+1, threshold=0.50, average='micro')
    
    file_path = f'./data/{model_iteration}/tfrecords/'
    
    # Model Inputs
    paper_title_input_ids = tf.keras.layers.Input((32,), dtype=tf.int64, name='paper_title_ids')
    abstract_input_ids = tf.keras.layers.Input((256,), dtype=tf.int64, name='abstract_ids')
    doc_type_id = tf.keras.layers.Input((1,), dtype=tf.int64, name='doc_type_id')
    journal_id = tf.keras.layers.Input((1,), dtype=tf.int64, name='journal_id')

    # Embedding Layers
    title_abstract_emb_layer = tf.keras.layers.Embedding(input_dim=len(title_vocab)+1, 
                                                 output_dim=int(args.emb_size), 
                                                 mask_zero=False, 
                                                 trainable=True,
                                                 name="title_embedding")
    
    title_position_embeddings = tf.keras.layers.Embedding(input_dim=32,
                                                          output_dim=int(args.emb_size),
                                                          weights=[get_pos_encoding_matrix(32, int(args.emb_size))],
                                                          name="title_position_embedding")(tf.range(start=0, 
                                                                                                    limit=32, delta=1))
    
    abstract_position_embeddings = tf.keras.layers.Embedding(input_dim=256,
                                                             output_dim=int(args.emb_size),
                                                             weights=[get_pos_encoding_matrix(256, int(args.emb_size))],
                                                             name="abs_position_embedding")(tf.range(start=0, 
                                                                                                     limit=256, delta=1))
    
    paper_title_embs = title_abstract_emb_layer(paper_title_input_ids)
    abstract_embs = title_abstract_emb_layer(abstract_input_ids)
    
    full_paper_title_embeddings = paper_title_embs + title_position_embeddings
    full_abstract_embeddings = abstract_embs + abstract_position_embeddings
    

    doc_embs = tf.keras.layers.Embedding(input_dim=len(doc_vocab)+1, 
                                         output_dim=int(args.emb_size), 
                                         mask_zero=False, 
                                         name="doc_type_embedding")(doc_type_id)

    journal_embs = tf.keras.layers.Embedding(input_dim=len(journal_vocab)+1, 
                                             output_dim=int(args.emb_size), 
                                             mask_zero=False, 
                                             name="journal_embedding")(journal_id)
    

    # First layer
    title_encoder_output = full_paper_title_embeddings
    abstract_encoder_output = full_abstract_embeddings
    
    for i in range(int(args.num_layers)):
        title_encoder_output = attention_module(title_encoder_output, title_encoder_output, title_encoder_output, 
                                                i, int(args.num_heads), int(args.emb_size), 'title')

    for i in range(int(args.num_layers)):
        abstract_encoder_output = attention_module(abstract_encoder_output, abstract_encoder_output, 
                                                   abstract_encoder_output, i, int(args.num_heads), int(args.emb_size),
                                                   'abstract')
    
    concat_output = tf.concat(values=[doc_embs, journal_embs, title_encoder_output, abstract_encoder_output,
                                      paper_title_embs, abstract_embs], axis=1)
        
    dense_output = tf.keras.layers.Dense(int(args.dense_1), activation='relu', 
                                         kernel_regularizer='L2', name="dense_1")(concat_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_1")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")(dense_output)
    

    # Second layer
    dense_output2 = tf.keras.layers.Dense(int(args.dense_2), activation='relu', 
                                         kernel_regularizer='L2', name="dense_2")(dense_output)
    dense_output2 = tf.keras.layers.Dropout(0.20, name="dropout_2")(dense_output2)
    dense_output2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")(dense_output2)
    dense_output_flat2 = tf.keras.layers.GlobalAveragePooling1D(name="title_pooling_layer2")(dense_output2)


    # Third Layer
    dense_output3 = tf.keras.layers.Dense(int(args.dense_3), activation='relu', 
                                         kernel_regularizer='L2', name="dense_3")(dense_output_flat2)
    dense_output3 = tf.keras.layers.Dropout(0.20, name="dropout_3")(dense_output3)
    dense_output3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_3")(dense_output3)

    class_prior = 1/len(target_vocab)
    last_layer_weight_init = tf.keras.initializers.Constant(class_prior)
    last_layer_bias_init = tf.keras.initializers.Constant(-np.log((1-class_prior)/class_prior))

    # Output Layer
    final_output = tf.keras.layers.Dense(len(target_vocab)+1, activation="sigmoid", 
                                         kernel_initializer=last_layer_weight_init,
                                         bias_initializer=last_layer_bias_init,
                                         name="cls")(dense_output3)

    mag_model = CustomModel(inputs=[paper_title_input_ids, abstract_input_ids, doc_type_id, journal_id], 
                            outputs=final_output, name='mag_model')
    
    if args.load_model == 'yes':
        saved_model = tf.keras.models.load_model(f'./models/{model_iteration}/{args.saved_model}')
        saved_model.save_weights(f'./models/{model_iteration}/{args.saved_model}_weights')
        mag_model.load_weights(f'./models/{model_iteration}/{model_iteration}_weights')
        
    optimizer = tf.keras.optimizers.Adam()

mag_model.compile(optimizer=optimizer)

curr_date = datetime.now().strftime("%Y%m%d")

filepath_1 = f"/home/ec2-user/Notebooks/models/{model_iteration}/" \
             f"{curr_date}_lr{args.learning_rate[2:]}_beta{args.beta.replace('.','')}" \
             f"_gamma{args.gamma.replace('.','')}_nH{str(args.num_heads)}_nL{str(args.num_layers)}" \
             f"_firstD{str(args.dense_1)}_secondD{str(args.dense_2)}_thirdD{str(args.dense_3)}/" \
    

filepath = filepath_1 + "model_epoch{epoch:02d}ckpt"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', 
                                                      verbose=0, save_best_only=False,
                                                      save_weights_only=False, mode='auto',
                                                      save_freq='epoch')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=8)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

callbacks = [model_checkpoint, early_stopping, lr_schedule]

train_ds = get_dataset(file_path, 'train')
val_ds = get_dataset(file_path, 'val')

mag_model.summary()
history = mag_model.fit(train_ds, epochs=int(args.epochs), validation_data=val_ds, verbose=1, callbacks=callbacks)

json.dump(str(history.history), open(f"{filepath_1}_{str(args.epochs)}EPOCHS_HISTORY.json", 'w+'))


