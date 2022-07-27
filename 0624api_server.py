#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
    api_server_flask.py
     sample api server by flask
'''
from flask import Flask, request, render_template, jsonify
import torch
import pandas as pd
from transformers import BertTokenizerFast, BertConfig, BertForTokenClassification
import dash_html_components as html
app = Flask(__name__)

@app.route('/')
def forms():
    return '<h1>Sample API Server</h1>\n'

CUDA_LAUNCH_BLOCKING = 1
ids_to_labels = {0: 'Ｘ', 1: 'Ｏ', -100:'IGN'}
labels_to_ids = {'Ｘ': 0, 'Ｏ': 1, 'IGN':-100}

ids_to_labels2 = {0: 'Ｘ', 1: 'Ｏ', 2: 'Ｖ', -100:'IGN'}
labels_to_ids2 = {'Ｘ': 0, 'Ｏ': 1, 'Ｖ':2,  'IGN':-100}
tokenizer = BertTokenizerFast.from_pretrained('cl-tohoku/bert-large-japanese')
device = torch.device("cpu")

model_V = BertForTokenClassification.from_pretrained('cl-tohoku/bert-large-japanese', num_labels=len(labels_to_ids))
model_V.load_state_dict(torch.load('model_V-Pla-present/pytorch_model.bin'))
model_Tari = BertForTokenClassification.from_pretrained('cl-tohoku/bert-large-japanese', num_labels=len(labels_to_ids))
model_Tari.load_state_dict(torch.load('model_Tari/pytorch_model.bin'))
model_Noni = BertForTokenClassification.from_pretrained('cl-tohoku/bert-large-japanese', num_labels=len(labels_to_ids2))      
model_Noni.load_state_dict(torch.load('model_Noni/pytorch_model.bin'))
model_IruAru = BertForTokenClassification.from_pretrained('cl-tohoku/bert-large-japanese', num_labels=len(labels_to_ids)) 
model_IruAru.load_state_dict(torch.load('model_IruAru/pytorch_model.bin'))
@app.route('/api/filter_single', methods=['GET','POST'])

def predict():
    article = request.json.get('article', 'This is the default article')
    print(article)
    pattern_id = request.json.get('pattern_id', 'V')
    print(pattern_id)
    article_list = [word for word in article]
    print(article_list)
    inputs = tokenizer(article_list,
                    is_pretokenized=True, 
                    return_offsets_mapping=True, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128,
                    return_tensors="pt")

    result_V, result_Tari, result_Noni, result_IruAru = '','','',''

    if 'V' in pattern_id:
        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        # forward pass
        outputs = model_V(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model_V.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue
        word_list = article_list.copy()
        for i in range(len(prediction)):
            word = prediction[i]          
            if word == 'Ｏ':
                word_list[i] = f'<span style="color:red;">{word_list[i]}</span>'
                
        prediction_V = ''.join(word_list)
        print(prediction)

        result_V = f'<span style="color:#C48888;">動詞普通型:</span><br/>{prediction_V}\n'

#==============================================================================================================
    if 'Tari' in pattern_id:
        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        # forward pass
        outputs = model_Tari(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model_Tari.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue

        word_list = article_list.copy()
        for i in range(len(prediction)):
            word = prediction[i]          
            if word == 'Ｏ':
                word_list[i] = f'<span style="color:red;">{word_list[i]}</span>'                

        prediction_Tari = ''.join(word_list)
        print(prediction)
        result_Tari = f'<span style="color:#C48888;">たり型:</span><br/>{prediction_Tari}\n'

#==============================================================================================================
    if 'Noni' in pattern_id:
        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        # forward pass
        outputs = model_Noni(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model_Noni.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [ids_to_labels2[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue

        word_list = article_list.copy()
        for i in range(len(prediction)):
            word = prediction[i]          
            if word == 'Ｏ':
                word_list[i] = f'<span style="color:red;">{word_list[i]}</span>'
            if word == 'Ｖ':
                word_list[i] = f'<span style="color:orange;">{word_list[i]}</span>'
                

        prediction_Noni = ''.join(word_list)
        print(prediction)
        print(article_list)

        result_Noni = f'<span style="color:#C48888;">のに型:</span><br/>{prediction_Noni}\n'
#==============================================================================================================
    if 'IruAru' in pattern_id:
        # move to gpu
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)

        # forward pass
        outputs = model_IruAru(ids, attention_mask=mask)
        logits = outputs[0]

        active_logits = logits.view(-1, model_IruAru.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        prediction = []
        for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        #only predictions on first word pieces are important
            if mapping[0] == 0 and mapping[1] != 0:
                prediction.append(token_pred[1])
            else:
                continue

        word_list = article_list.copy()
        for i in range(len(prediction)):
            word = prediction[i]          
            if word == 'Ｏ':
                word_list[i] = f'<span style="color:red;">{word_list[i]}</span>'
                

        prediction_IruAru = ''.join(word_list)
        print(prediction)
        print(article_list)
        result_IruAru = f'<span style="color:#C48888;">いる/ある型:</span><br/>{prediction_IruAru}\n'
#==============================================================================================================
	
    result_json = {
    'pattern_id' : pattern_id,
    'html_V': result_V,
    'html_Tari': result_Tari,
    'html_Noni': result_Noni,
    'html_IruAru': result_IruAru,
    }
    return jsonify(result_json)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30501)


# In[ ]:




