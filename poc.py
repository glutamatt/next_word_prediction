# %%

import sys
import torch
import string

#from transformers import BertTokenizer, BertForMaskedLM
#bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()


from transformers import FlaubertTokenizer, FlaubertWithLMHeadModel
flaubert_tokenizer = FlaubertTokenizer.from_pretrained('flaubert/flaubert_large_cased')
flaubert_model = FlaubertWithLMHeadModel.from_pretrained('flaubert/flaubert_large_cased')

def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
    if text_sentence.find("<mask>") == -1:
        text_sentence = text_sentence + " <mask>"

    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx, text_sentence


def get_all_predictions(text_sentence, top_clean=5):
    input_ids, mask_idx, text_sentence  = encode(flaubert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = flaubert_model(input_ids)[0]
    flaubert = decode(flaubert_tokenizer, predict[0, mask_idx, :].topk(top_clean).indices.tolist(), top_clean)

    return flaubert.splitlines(), text_sentence

for name in map(str.rstrip, sys.stdin):
    preds, text_sentence = get_all_predictions(name, 50)
    for p in preds:
        print(text_sentence.replace(flaubert_tokenizer.mask_token, p))
