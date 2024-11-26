import json
import pandas as pd
import matplotlib.pyplot as plt
import re 
import numpy as np
from transformers import BertModel
from transformers import AutoTokenizer


import torch


# Notice there is a single TODO in the model
class CauseDetector(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embeddings_tensor: torch.FloatTensor,
        pad_idx: int,
        output_size: int,
        dropout_val: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx ## what is the pad_idx
        self.dropout_val = dropout_val
        self.output_size = output_size
        # TODO: Initialize the embeddings from the weights matrix.
        #       Check the documentation for how to initialize an embedding layer
        #       from a pretrained embedding matrix. 
        #       Be careful to set the `freeze` parameter!
        #       Docs are here: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings_tensor, freeze = True)
        # Dropout regularization
        # https://jmlr.org/papers/v15/srivastava14a.html
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_val, inplace=False)
        # Bidirectional 2-layer LSTM. Feel free to try different parameters.
        # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
        self.lstm = torch.nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=2,
            dropout=dropout_val,
            batch_first=True,
            bidirectional=True,
        )
        # For classification over the final LSTM state.
        self.classifier = torch.nn.Linear(hidden_dim*2, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
    
    def encode_text(
        self,
        symbols: torch.Tensor
    ) -> torch.Tensor:
        """Encode the (batch of) sequence(s) of token symbols with an LSTM.
            Then, get the last (non-padded) hidden state for each symbol and return that.

        Args:
            symbols (torch.Tensor): The batch size x sequence length tensor of input tokens

        Returns:
            torch.Tensor: The final hiddens tate of the LSTM, which represents an encoding of
                the entire sentence
        """
        # First we get the embedding for each input symbol
        #print(self.embeddings)
        embedded = self.embeddings(symbols)
        embedded = self.dropout_layer(embedded)
        # Packs embedded source symbols into a PackedSequence.
        # This is an optimization when using padded sequences with an LSTM
        #print(symbols)
        #print((symbols != self.pad_idx))
        #print(self.pad_idx)
        lens = (symbols != self.pad_idx).sum(dim=1).to("cpu")
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lens, batch_first=True, enforce_sorted=False
        )
        # -> batch_size x seq_len x encoder_dim, (h0, c0).
        packed_outs, (H, C) = self.lstm(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Now we have the representation of eahc token encoded by the LSTM.
        encoded, (H, C) = self.lstm(embedded)
        
        # This part looks tricky. All we are doing is getting a tensor
        # That indexes the last non-PAD position in each tensor in the batch.
        last_enc_out_idxs = lens - 1
        # -> B x 1 x 1.
        last_enc_out_idxs = last_enc_out_idxs.view([encoded.size(0)] + [1, 1])
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_enc_out_idxs = last_enc_out_idxs.expand(
            [-1, -1, encoded.size(-1)]
        )
        # Get the final hidden state in the LSTM
        last_hidden = torch.gather(encoded, 1, last_enc_out_idxs)
        return last_hidden
    
    def forward(
        self,
        symbols: torch.Tensor,
    ) -> torch.Tensor:
        encoded_sents = self.encode_text(symbols)
        output = self.classifier(encoded_sents)
        return self.log_softmax(output)



def get_target_conv_utt_ids(dia_utt, pattern = r'\d+' ):
    # dia_utt is the dialogu we are looking at with the targeted utterances 'dia1utt3'
    target_ids = re.findall(pattern,dia_utt)
    return target_ids[0], target_ids[1]

def get_cause_span_ids(cause_spans):
    ids = []
    for span in cause_spans:
        span_as_list = span.split('_')
        utterance_ID = span_as_list[0]
        ids.append(int(utterance_ID))
    return ids

def count_labels(cause_pair):
    count = 0
    for dp in cause_pair:
        if dp['label'] == 1:
            count += 1
    return count


def get_train_val_test(data_pairs, train_split = .6 , val_split = .2, test_split = .2):
    dpl = len(data_pairs)
    train = data_pairs[:int(dpl * train_split)]
    val = data_pairs[int(dpl * train_split):int(dpl * train_split) + int(val_split * dpl)]
    test = data_pairs[int(dpl * train_split) + int(val_split * dpl):]
    return train, val, test

def load_data():
   
    f = open('data/Subtask_1_1_train.json')

    data = json.load(f)

    f.close()

    return data

def format_data(data):
    utt_emotions, cause_spans, utterances, speakers, emotions, dia_utt, all_vocab = [], [], [], [], [], [], []
    pattern = r'\d+'
    
    for d in data:
        matches = re.findall(pattern,d['emotion_utterance_ID'], flags=0)
        dia, utt = matches[0], matches[1]
        dia_utt.append([dia,utt])
        utt_emotions.append(d['emotion'])
        cause_spans.append(np.array(d['cause_spans']))
        c = []
        s = []
        e = []
        
        for convo in d['conversation']: # [:int(utt) ]:
            all_vocab.append(convo['text'])
            c.append(convo['text'])
            s.append(convo['speaker'])
            e.append(convo['emotion'])
        utterances.append(c)
        speakers.append(s)
        emotions.append(e)
    
    utt_emotions = np.array(utt_emotions)

    return utt_emotions, cause_spans, utterances, speakers, emotions, dia_utt, all_vocab

from typing import Dict, List, Optional, Tuple
from collections import Counter

import torch
import numpy as np
import spacy

class Tokenizer:
    """Tokenizes and pads a batch of input sentences."""
    def __init__(self, pad_symbol: Optional[str] = "<PAD>"):
        """Initializes the tokenizer
        Args:
            pad_symbol (Optional[str], optional): The symbol for a pad. Defaults to "<PAD>".
        """
        self.pad_symbol = pad_symbol
        self.nlp = spacy.load("en_core_web_sm")
    def __call__(self, batch: List[str]) -> List[List[str]]:
        """Tokenizes each sentence in the batch, and pads them if necessary so
        that we have equal length sentences in the batch.
        Args:
            batch (List[str]): A List of sentence strings
        Returns:
            List[List[str]]: A List of equal-length token Lists.
        """
        batch = self.tokenize(batch)
        batch = self.pad(batch)
        return batch

    def tokenize(self, sentences: List[str]) -> List[List[str]]:
        """Tokenizes the List of string sentences into a Lists of tokens using spacy tokenizer.

        Args:
            sentences (List[str]): The input sentence.

        Returns:
            List[str]: The tokenized version of the sentence.
        """
        tokened_sents = []
        for sent in sentences:
            tokened_sents.append(['<SOS>'] + [w.text for w in self.nlp(sent)] + ['<EOS>'])
        return tokened_sents

    def pad(self, batch: List[List[str]]) -> List[List[str]]:
        """Appends pad symbols to each tokenized sentence in the batch such that
        every List of tokens is the same length. This means that the max length sentence
        will not be padded.

        Args:
            batch (List[List[str]]): Batch of tokenized sentences.

        Returns:
            List[List[str]]: Batch of padded tokenized sentences. 
        """
        max_len = len(max(batch, key=len))
        for sent in batch:
            pad_len = max_len - len(sent) 
            sent += ['<P>'] * pad_len
        # TODO: For each sentence in the batch, append the special <P>
        #       symbol to it n times to make all sentences equal length
        return batch



# Nothing to do for this class!

class BatchTokenizer:
    """Tokenizes and pads a batch of input sentences."""

    def __init__(self, model_name='prajjwal1/bert-small'):
        """Initializes the tokenizer

        Args:
            pad_symbol (Optional[str], optional): The symbol for a pad. Defaults to "<P>".
        """
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def get_sep_token(self,):
        return self.hf_tokenizer.sep_token

    def __call__(self, prem_batch: List[str], hyp_batch: List[str]) -> List[List[str]]:
        """Uses the huggingface tokenizer to tokenize and pad a batch.

        We return a dictionary of tensors per the huggingface model specification.

        Args:
            batch (List[str]): A List of sentence strings

        Returns:
            Dict: The dictionary of token specifications provided by HuggingFace
        """
        # The HF tokenizer will PAD for us, and additionally combine
        # The two sentences deimited by the [SEP] token.
        enc = self.hf_tokenizer(
            prem_batch,
            hyp_batch,
            padding=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        return enc


def generate_pairwise_input(dataset):
    """
    TODO: group all premises and corresponding hypotheses and labels of the datapoints
    a datapoint as seen earlier is a dict of premis, hypothesis and label
    """
    pos_causes = []
    targets = []
    labels = []
    for x in dataset:
        pos_causes.append(x['pos_cause'])
        targets.append(x['target'])
        labels.append(x['label'])

    return pos_causes, targets, labels


def chunk(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        # yeild function is similair to the return function, but it continues to give the value back. here are are making batches
        yield lst[i:i + n]

def chunk_multi(lst1, lst2, n):
    for i in range(0, len(lst1), n):
        yield lst1[i: i + n], lst2[i: i + n]


def encode_labels(labels):
    """Turns the batch of labels into a tensor

    Args:
        labels (List[int]): List of all labels in the batch

    Returns:
        torch.FloatTensor: Tensor of all labels in the batch
    """
    return torch.LongTensor([int(l) for l in labels])
    