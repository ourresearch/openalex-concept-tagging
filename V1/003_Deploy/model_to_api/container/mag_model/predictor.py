# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

import os
import json
import flask
import pickle
import pandas as pd
import tensorflow as tf

# Define the path
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

# Load the dictionaries
with open(os.path.join(model_path, "topics_vocab.pkl"), "rb") as f:
    target_vocab = pickle.load(f)
    
target_vocab_inv = {j:i for i,j in target_vocab.items()}

print("Loaded target vocab")

with open(os.path.join(model_path, "doc_type_vocab.pkl"), "rb") as f:
    doc_vocab = pickle.load(f)
    
doc_vocab_inv = {j:i for i,j in doc_vocab.items()}

print("Loaded doc_type vocab")

with open(os.path.join(model_path, "journal_name_vocab.pkl"), "rb") as f:
    journal_vocab = pickle.load(f)
    
journal_vocab_inv = {j:i for i,j in journal_vocab.items()}

print("Loaded journal vocab")

with open(os.path.join(model_path, "paper_title_vocab.pkl"), "rb") as f:
    title_vocab = pickle.load(f)
    
title_vocab_inv = {j:i for i,j in title_vocab.items()}

print("Loaded title vocab")

with open(os.path.join(model_path, "tag_id_vocab.pkl"), "rb") as f:
    tag_id_vocab = pickle.load(f)

print("Loaded tag ID vocab")

encoding_layer = tf.keras.layers.experimental.preprocessing.CategoryEncoding(
    max_tokens=len(target_vocab)+1, output_mode="binary", sparse=False)

# Load the model components
raw_model = tf.keras.models.load_model(os.path.join(model_path, 'mag_model_500_basic'), compile=False)
raw_model.trainable = False

print("Loaded raw model")

mag_model = tf.keras.Model(inputs=raw_model.inputs, 
                           outputs=tf.math.top_k(raw_model.outputs, k=25))

print("Created full model")

def tokenize_feature(feature, feature_name='doc_type'):
    if feature_name=='doc_type':
        vocab = doc_vocab
    else:
        vocab = journal_vocab
    unk_token_id = vocab.get('[UNK]')
    none_token_id = vocab.get('[NONE]')
    if feature:
        token_feature = [vocab.get(feature, unk_token_id)]
    else:
        token_feature = [none_token_id]
    return token_feature

def tokenize_title(feature):
    split_feature = feature.split(" ")
    vocab = title_vocab
    unk_token_id = vocab.get('[UNK]')
    none_token_id = vocab.get('[NONE]')
    if feature:
        token_feature = [vocab.get(x, unk_token_id) for x in split_feature]
    else:
        token_feature = [none_token_id]
    return token_feature

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        _ = mag_model.get_layer('cls')
        status = 200
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input_json = json.dumps(input_json)
    input_df = pd.read_json(input_json, orient='records').reset_index()

    # Tokenize data
    input_df['title'] = input_df['title'].apply(lambda x: x.lower().strip())
    input_df['paper_title_tok'] = input_df['title'].apply(tokenize_title)
    input_df['doc_type_tok'] = input_df['doc_type'].apply(tokenize_feature, args=('doc_type',))
    input_df['journal_tok'] = input_df['journal'].apply(tokenize_feature, args=('journal',))
    
    paper_titles = tf.keras.preprocessing.sequence.pad_sequences(input_df['paper_title_tok'].to_list(), maxlen=64, 
                                                             dtype='int64', padding='post', 
                                                             truncating='post', value=0)
    
    doc_types = tf.convert_to_tensor(input_df['doc_type_tok'].to_list())
    journal = tf.convert_to_tensor(input_df['journal_tok'].to_list())
    
    # Predict
    model_output = mag_model([paper_titles, doc_types, journal])
    
    scores = model_output.values.numpy()[0].tolist()
    preds = model_output.indices.numpy()[0].tolist()

    # Transform predicted labels into tags
    all_tags = []
    for score, pred in zip(scores, preds):
        tags = []
        scores = []
        tag_ids = []
        for i in range(25):
            if score[i] >= 0.32:
                tags.append(target_vocab_inv.get(pred[i]))
                scores.append(score[i])
                tag_ids.append(tag_id_vocab.get(pred[i]))
        all_tags.append({"tags": tags, "scores": scores, "tag_ids": tag_ids})

    # Transform predictions to JSON
    result = json.dumps(all_tags)
    return flask.Response(response=result, status=200, mimetype='application/json')
