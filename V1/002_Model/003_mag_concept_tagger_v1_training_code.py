import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import os
import json
import argparse
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from typeguard import typechecked
from tensorflow_addons.utils.keras_utils import LossFunctionWrapper
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
from datetime import datetime


class SigmoidFocalCrossEntropy(LossFunctionWrapper):
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf).

    Args:
      alpha: balancing factor, default value is 0.25.
      gamma: modulating factor, default value is 2.0.
    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
          shape as `y_true`; otherwise, it is scalar.
    Raises:
        ValueError: If the shape of `sample_weight` is invalid or value of
          `gamma` is less than zero.
    """

    @typechecked
    def __init__(
        self,
        from_logits: bool = False,
        alpha: FloatTensorLike = 0.25,
        gamma: FloatTensorLike = 2.0,
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "sigmoid_focal_crossentropy",
    ):
        super().__init__(
            sigmoid_focal_crossentropy,
            name=name,
            reduction=reduction,
            from_logits=from_logits,
            alpha=alpha,
            gamma=gamma,
        )


@tf.function
def sigmoid_focal_crossentropy(y_true: TensorLike,
                               y_pred: TensorLike,
                               alpha: FloatTensorLike = 0.25,
                               gamma: FloatTensorLike = 2.0,
                               from_logits: bool = False,) -> tf.Tensor:
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf).

    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero.")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, dtype=y_pred.dtype)

    # Get the cross_entropy for each entry
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha is not None:
        alpha = tf.cast(alpha, dtype=y_true.dtype)
        alpha_factor = tf.math.multiply(y_true, alpha) + tf.math.multiply((1 - y_true), (1 - alpha))

    if gamma:
        gamma = tf.cast(gamma, dtype=y_true.dtype)
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(tf.math.multiply(alpha_factor, ce) * modulating_factor, axis=-1)


class CustomModel(tf.keras.Model):
    def train_step(self, inputs):
        old_features, labels = inputs
        labels = tf.RaggedTensor.from_tensor(labels, padding=0)
        paper_titles = old_features[0][:,:64].to_tensor(shape=[None, 64])

        features = (paper_titles, old_features[-2], old_features[-1])
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
        paper_titles = old_features[0][:,:64].to_tensor(shape=[None, 64])

        features = (paper_titles, old_features[-2], old_features[-1])
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
        'journal': tf.io.FixedLenFeature((1,), tf.int64),
        'doc_type': tf.io.FixedLenFeature((1,), tf.int64),
        'targets': tf.io.FixedLenFeature((20,), tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    paper_title = example['paper_title']
    doc_type = example['doc_type']
    journal = example['journal']
    targets = example['targets']

    return (paper_title, doc_type, journal), targets


def attention_module(query, key, value, i, num_heads, emb_dim=512):
    # Multi headed self-attention
    attention_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=emb_dim // num_heads,
        name="encoder_{}/multiheadattention".format(i),
    )(query, key, value)
    attention_output = tf.keras.layers.Dropout(0.1, name="encoder_{}/att_dropout".format(i))(
        attention_output
    )
    attention_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/att_layernormalization".format(i)
    )(query + attention_output)

    # Feed-forward layer
    ffn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(emb_dim),
        ],
        name="encoder_{}/ffn".format(i),
    )
    ffn_output = ffn(attention_output)
    ffn_output = tf.keras.layers.Dropout(0.1, name="encoder_{}/ffn_dropout".format(i))(
        ffn_output
    )
    sequence_output = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="encoder_{}/ffn_layernormalization".format(i)
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


def get_class_weights(df, beta):
    class_counts = df[df['count'] > 4]['count'].to_list()
    if beta == 1.0:
        weights = 1/np.array(class_counts)
    else:
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.array(effective_num)
    weights = (weights/np.sum(weights)).tolist()
    final_class_weights = np.array([0.0] + weights)
    return final_class_weights


def get_dataset(path, data_type='train'):

    tfrecords = [f"{path}{data_type}/{x}" for x in os.listdir(f"{path}{data_type}/") if x.endswith('tfrecord')]
    tfrecords.sort()


    raw_dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO)

    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTO)

    parsed_dataset = parsed_dataset.apply(tf.data.experimental.dense_to_ragged_batch(1024,drop_remainder=True))
    return parsed_dataset


parser = argparse.ArgumentParser(description='parsing arguments to set config')

parser.add_argument('-e', '--epochs', action='store', dest='epochs', default=20)
parser.add_argument('-lr', '--learningRate', action='store', dest='learning_rate', default='0.0001')
parser.add_argument('-g', '--gamma', action='store', dest='gamma', default='2.0')
parser.add_argument('-b', '--beta', action='store', dest='beta', default='0.90')
parser.add_argument('-lm', '--loadModel', action='store', dest='load_model', default='no')
parser.add_argument('-sm', '--savedModel', action='store', dest='saved_model', default='none')
parser.add_argument('-nl', '--numLayers', action='store', dest='num_layers', default='1')
parser.add_argument('-nh', '--numHeads', action='store', dest='num_heads', default='4')

args = parser.parse_args()


mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with mirrored_strategy.scope():
    AUTO = tf.data.experimental.AUTOTUNE

    model_iteration = 'iteration_final/basic_word_tokenized'


    with open(f"./{model_iteration}/vocab/topics_vocab.pkl", "rb") as f:
        target_vocab = pickle.load(f)

    with open(f"./{model_iteration}/vocab/doc_type_vocab.pkl", "rb") as f:
        doc_vocab = pickle.load(f)

    with open(f"./{model_iteration}/vocab/journal_name_vocab.pkl", "rb") as f:
        journal_vocab = pickle.load(f)

    with open(f"./{model_iteration}/vocab/paper_title_vocab.pkl", "rb") as f:
        title_vocab = pickle.load(f)

    print(f"Topic vocab len: {len(target_vocab)}")


    encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
        max_tokens=len(target_vocab)+1, output_mode="binary", sparse=False)

    counts_df = pd.read_pickle(f"./{model_iteration}/vocab/topics_vocab_counts.pkl")

    class_weights = get_class_weights(counts_df, float(args.beta))

    loss_fn = tfa.losses.SigmoidFocalCrossEntropy(alpha=float(args.beta), gamma=float(args.gamma),
                                       reduction=tf.keras.losses.Reduction.NONE)

    metric_1 = tf.keras.metrics.CategoricalAccuracy()
    metric_2 = tf.keras.metrics.Recall(thresholds=0.50)
    metric_3 = tf.keras.metrics.Precision(thresholds=0.50)
    metric_4 = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
    metric_5 = tfa.metrics.FBetaScore(num_classes=len(target_vocab)+1, beta=0.5, threshold=0.50, average='micro')
    metric_6 = tfa.metrics.F1Score(num_classes=len(target_vocab)+1, threshold=0.50, average='micro')

    file_path = f'./{model_iteration}/tfrecords/'

    # Model Inputs
    paper_title_input_ids = tf.keras.layers.Input((64,), dtype=tf.int64, name='paper_title_ids')
    doc_type_id = tf.keras.layers.Input((1,), dtype=tf.int64, name='doc_type_id')
    journal_id = tf.keras.layers.Input((1,), dtype=tf.int64, name='journal_id')

    # Embedding Layers
    paper_title_embs = tf.keras.layers.Embedding(input_dim=len(title_vocab)+1,
                                                 output_dim=512,
                                                 mask_zero=False,
                                                 trainable=True,
                                                 name="title_embedding")(paper_title_input_ids)

    position_embeddings = tf.keras.layers.Embedding(input_dim=64,
                                                    output_dim=512,
                                                    weights=[get_pos_encoding_matrix(64, 512)],
                                                    name="position_embedding")(tf.range(start=0, limit=64, delta=1))

    embeddings = paper_title_embs + position_embeddings


    doc_embs = tf.keras.layers.Embedding(input_dim=len(doc_vocab)+1,
                                         output_dim=32,
                                         mask_zero=False,
                                         name="doc_type_embedding")(doc_type_id)

    journal_embs = tf.keras.layers.Embedding(input_dim=len(journal_vocab)+1,
                                             output_dim=128,
                                             mask_zero=False,
                                             name="journal_embedding")(journal_id)

    # First layer
    encoder_output = embeddings

    for i in range(int(args.num_layers)):
        encoder_output = attention_module(encoder_output, encoder_output, encoder_output, i, int(args.num_heads))
    dense_output = tf.keras.layers.Dense(2048, activation='relu',
                                         kernel_regularizer='L2', name="dense_1")(encoder_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_1")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")(dense_output)
    dense_output_flat = tf.keras.layers.GlobalAveragePooling1D(name="title_pooling_layer")(dense_output)
    doc_flat = tf.keras.layers.GlobalAveragePooling1D(name="doc_pooling_layer")(doc_embs)
    journal_flat = tf.keras.layers.GlobalAveragePooling1D(name="journal_pooling_layer")(journal_embs)

    # Second layer
    dense_output2 = tf.keras.layers.Dense(1024, activation='relu',
                                         kernel_regularizer='L2', name="dense_2")(paper_title_embs)
    dense_output2 = tf.keras.layers.Dropout(0.20, name="dropout_2")(dense_output2)
    dense_output2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")(dense_output2)
    dense_output_flat2 = tf.keras.layers.GlobalAveragePooling1D(name="title_pooling_layer2")(dense_output2)
    concat_output = tf.concat(values=[dense_output_flat, dense_output_flat2, journal_flat, doc_flat], axis=1)

    # Third Layer
    dense_output = tf.keras.layers.Dense(1024, activation='relu',
                                         kernel_regularizer='L2', name="dense_3")(concat_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_3")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_3")(dense_output)

    class_prior = 1/len(target_vocab)
    last_layer_weight_init = tf.keras.initializers.Constant(class_prior)
    last_layer_bias_init = tf.keras.initializers.Constant(-np.log((1-class_prior)/class_prior))

    # Output Layer
    final_output = tf.keras.layers.Dense(len(target_vocab)+1, activation="sigmoid",
                                         kernel_initializer=last_layer_weight_init,
                                         bias_initializer=last_layer_bias_init,
                                         name="cls")(dense_output)

    mag_model = CustomModel(inputs=[paper_title_input_ids, doc_type_id, journal_id],
                            outputs=final_output, name='mag_model')

    if args.load_model == 'yes':
        saved_model = tf.keras.models.load_model(f'./models/{model_iteration}/{args.saved_model}')
        saved_model.save_weights(f'./models/{model_iteration}/{args.saved_model}_weights')
        mag_model.load_weights(f'./models/{model_iteration}/{model_iteration}_weights')

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))

mag_model.compile(optimizer=optimizer)

curr_date = datetime.now().strftime("%Y%m%d")

filepath_1 = f"/home/ec2-user/Notebooks/models/{model_iteration}/partial_model{curr_date}_lr{args.learning_rate[2:]}_beta{args.beta.replace('.','')}" \
             f"_gamma{args.gamma.replace('.','')}_nH{str(args.num_heads)}_nL{str(args.num_layers)}/"

filepath = filepath_1 + "model_epoch{epoch:02d}ckpt"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                      verbose=0, save_best_only=False,
                                                      save_weights_only=False, mode='auto',
                                                      save_freq='epoch')

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=8)

callbacks = [model_checkpoint, early_stopping]

train_ds = get_dataset(file_path, 'train')
val_ds = get_dataset(file_path, 'val')

mag_model.summary()
history = mag_model.fit(train_ds, epochs=int(args.epochs), validation_data=val_ds, verbose=1, callbacks=callbacks)

json.dump(history.history, open(f"{filepath_1}_{str(args.epochs)}EPOCHS_HISTORY.json", 'w+'))
