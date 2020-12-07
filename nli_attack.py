import sys
import pickle
import argparse
import os
from pathlib import Path
import numpy as np
np.random.seed(1234)
from scipy.special import softmax
import fnmatch
import criteria
import string
import pickle
import random
random.seed(0)
import csv
from fuzzywuzzy import fuzz
from InferSent.models import NLINet, InferSent
from esim.model import ESIM
from esim.data import Preprocessor
from esim.utils import correct_predictions
from collections import defaultdict
import tensorflow.compat.v1 as tf
#To make tf 2.0 compatible with tf1.0 code, we disable the tf2.0 functionalities
tf.compat.v1.disable_eager_execution()
import tensorflow_hub as hub
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, TensorDataset
from BERT.tokenization import BertTokenizer
from BERT.modeling import BertForSequenceClassification, BertConfig


class NLI_infer_InferSent(nn.Module):
    def __init__(self,
                 pretrained_file,
                 embedding_path,
                 data,
                 batch_size=32):
        super(NLI_infer_InferSent, self).__init__()

        #         self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cpu")
        # torch.cuda.set_device(local_rank)

        # Retrieving model parameters from checkpoint.
        config_nli_model = {
            'word_emb_dim': 300,
            'enc_lstm_dim': 2048,
            'n_enc_layers': 1,
            'dpout_model': 0.,
            'dpout_fc': 0.,
            'fc_dim': 512,
            'bsize': batch_size,
            'n_classes': 3,
            'pool_type': 'max',
            'nonlinear_fc': 0,
            'encoder_type': 'InferSent',
            'use_cuda': True,
            'use_target': False,
            'version': 1,
        }
        params_model = {'bsize': 64, 'word_emb_dim': 200, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}

        print("\t* Building model...")
        self.model = NLINet(config_nli_model).cuda()
        print("Reloading pretrained parameters...")
        self.model.load_state_dict(torch.load(os.path.join("savedir/", "model.pickle")))

        # construct dataset loader
        print('Building vocab and embeddings...')
        self.dataset = NLIDataset_InferSent(embedding_path, data=data, batch_size=batch_size)
    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        data_batches = self.dataset.transform_text(text_data)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in data_batches:
                # Move input and output data to the GPU if one is used.
                (s1_batch, s1_len), (s2_batch, s2_len) = batch
                s1_batch, s2_batch = s1_batch.cuda(), s2_batch.cuda()
                logits = self.model((s1_batch, s1_len), (s2_batch, s2_len))
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)

class NLI_infer_ESIM(nn.Module):
    def __init__(self,
                 pretrained_file,
                 worddict_path,
                 local_rank=-1,
                 batch_size=32):
        super(NLI_infer_ESIM, self).__init__()

        self.batch_size = batch_size
        self.device = torch.device("cuda:{}".format(local_rank) if local_rank > -1 else "cuda")
        checkpoint = torch.load(pretrained_file)
        # Retrieving model parameters from checkpoint.
        vocab_size = checkpoint['model']['_word_embedding.weight'].size(0)
        embedding_dim = checkpoint['model']['_word_embedding.weight'].size(1)
        hidden_size = checkpoint['model']['_projection.0.weight'].size(0)
        num_classes = checkpoint['model']['_classification.4.weight'].size(0)

        print("\t* Building model...")
        self.model = ESIM(vocab_size,
                          embedding_dim,
                          hidden_size,
                          num_classes=num_classes,
                          device=self.device).to(self.device)

        self.model.load_state_dict(checkpoint['model'])

        # construct dataset loader
        self.dataset = NLIDataset_ESIM(worddict_path)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()
        device = self.device

        # transform text data into indices and create batches
        self.dataset.transform_text(text_data)
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)

        # Deactivate autograd for evaluation.
        probs_all = []
        with torch.no_grad():
            for batch in dataloader:
                # Move input and output data to the GPU if one is used.
                premises = batch['premise'].to(device)
                premises_lengths = batch['premise_length'].to(device)
                hypotheses = batch['hypothesis'].to(device)
                hypotheses_lengths = batch['hypothesis_length'].to(device)

                _, probs = self.model(premises,
                                      premises_lengths,
                                      hypotheses,
                                      hypotheses_lengths)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class NLI_infer_BERT(nn.Module):
    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        super(NLI_infer_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(pretrained_dir, num_labels=3).cuda()

        # construct dataset loader
        self.dataset = NLIDataset_BERT(pretrained_dir, max_seq_length=max_seq_length, batch_size=batch_size)

    def text_pred(self, text_data):
        # Switch the model to eval mode.
        self.model.eval()

        # transform text data into indices and create batches
        dataloader = self.dataset.transform_text(text_data)

        probs_all = []
        #         for input_ids, input_mask, segment_ids in tqdm(dataloader, desc="Evaluating"):
        for input_ids, input_mask, segment_ids in dataloader:
            input_ids = input_ids.cuda()
            input_mask = input_mask.cuda()
            segment_ids = segment_ids.cuda()

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)
                probs = nn.functional.softmax(logits, dim=-1)
                probs_all.append(probs)

        return torch.cat(probs_all, dim=0)


class USE(object):
    def __init__(self, cache_path):
        super(USE, self).__init__()
        os.environ['TFHUB_CACHE_DIR'] = cache_path
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
        self.embed = hub.Module(module_url)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.build_graph()
        self.sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def build_graph(self):
        self.sts_input1 = tf.placeholder(tf.string, shape=(None))
        self.sts_input2 = tf.placeholder(tf.string, shape=(None))

        sts_encode1 = tf.nn.l2_normalize(self.embed(self.sts_input1), axis=1)
        sts_encode2 = tf.nn.l2_normalize(self.embed(self.sts_input2), axis=1)
        self.cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
        clip_cosine_similarities = tf.clip_by_value(self.cosine_similarities, -1.0, 1.0)
        self.sim_scores = 1.0 - tf.acos(clip_cosine_similarities)

    def semantic_sim(self, sents1, sents2):
        scores = self.sess.run(
            [self.sim_scores],
            feed_dict={
                self.sts_input1: sents1,
                self.sts_input2: sents2,
            })
        return scores


def pick_most_similar_words_batch(src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
    """
    embeddings is a matrix with (d, vocab_size)
    """
    sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
    sim_words, sim_values = [], []
    for idx, src_word in enumerate(src_words):
        sim_value = sim_mat[src_word][sim_order[idx]]
        mask = sim_value >= threshold
        sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
        sim_word = [idx2word[id] for id in sim_word]
        sim_words.append(sim_word)
        sim_values.append(sim_value)
    return sim_words, sim_values


def read_data(filepath, data_size, target_model='infersent', lowercase=False, ignore_punctuation=False, stopwords=[]):
    """
    Read the premises, hypotheses and labels from some NLI dataset's
    file and return them in a dictionary. The file should be in the same
    form as SNLI's .txt files.

    Args:
        filepath: The path to a file containing some premises, hypotheses
            and labels that must be read. The file should be formatted in
            the same way as the SNLI (and MultiNLI) dataset.

    Returns:
        A dictionary containing three lists, one for the premises, one for
        the hypotheses, and one for the labels in the input data.
    """
    if target_model == 'bert':
        labeldict = {"contradiction": 0,
                      "entailment": 1,
                      "neutral": 2}
    else:
        labeldict = {"entailment": 0,
                     "neutral": 1,
                     "contradiction": 2}
    with open(filepath, 'r', encoding='utf8') as input_data:
        premises, hypotheses, labels = [], [], []

        # Translation tables to remove punctuation from strings.
        punct_table = str.maketrans({key: ' '
                                     for key in string.punctuation})

        for idx, line in enumerate(input_data):
            if idx >= data_size:
                break

            line = line.strip().split('\t')

            # Ignore sentences that have no gold label.
            if line[0] == '-':
                continue

            premise = line[1]
            hypothesis = line[2]

            if lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in stopwords])
            labels.append(labeldict[line[0]])

        return {"premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


class NLIDataset_ESIM(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 worddict_path,
                 padding_idx=0,
                 bos="_BOS_",
                 eos="_EOS_"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.padding_idx = padding_idx

        # build word dict
        with open(worddict_path, 'rb') as pkl:
            self.worddict = pickle.load(pkl)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {
            "premise": self.data["premises"][index],
            "premise_length": min(self.premises_lengths[index],
                                  self.max_premise_length),
            "hypothesis": self.data["hypotheses"][index],
            "hypothesis_length": min(self.hypotheses_lengths[index],
                                     self.max_hypothesis_length)
        }

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict['_OOV_']
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"premises": [],
                            "hypotheses": []}

        for i, premise in enumerate(data['premises']):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def transform_text(self, data):
        #         # standardize data format
        #         data = defaultdict(list)
        #         for hypothesis in hypotheses:
        #             data['premises'].append(premise)
        #             data['hypotheses'].append(hypothesis)

        # transform data into indices
        data = self.transform_to_indices(data)

        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {
            "premises": torch.ones((self.num_sequences,
                                    self.max_premise_length),
                                   dtype=torch.long) * self.padding_idx,
            "hypotheses": torch.ones((self.num_sequences,
                                      self.max_hypothesis_length),
                                     dtype=torch.long) * self.padding_idx}

        for i, premise in enumerate(data["premises"]):
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])



class NLIDataset_InferSent(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 embedding_path,
                 data,
                 word_emb_dim=300,
                 batch_size=32,
                 bos="<s>",
                 eos="</s>"):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.bos = bos
        self.eos = eos
        self.word_emb_dim = word_emb_dim
        self.batch_size = batch_size

        # build word dict
        self.word_vec = self.build_vocab(data['premises']+data['hypotheses'], embedding_path)

    def build_vocab(self, sentences, embedding_path):
        word_dict = self.get_word_dict(sentences)
        word_vec = self.get_embedding(word_dict, embedding_path)
        print('Vocab size : {0}'.format(len(word_vec)))
        return word_vec

    def get_word_dict(self, sentences):
        # create vocab of words
        word_dict = {}
        for sent in sentences:
            for word in sent:
                if word not in word_dict:
                    word_dict[word] = ''
        word_dict['<s>'] = ''
        word_dict['</s>'] = ''
        word_dict['<oov>'] = ''
        return word_dict

    def get_embedding(self, word_dict, embedding_path):
        # create word_vec with glove vectors
        word_vec = {}
        word_vec['<oov>'] = np.random.normal(size=(self.word_emb_dim))
        with open(embedding_path) as f:
            for line in f:
                word, vec = line.split(' ', 1)
                if word in word_dict:
                    word_vec[word] = np.array(list(map(float, vec.split())))
        print('Found {0}(/{1}) words with embedding vectors'.format(
            len(word_vec), len(word_dict)))
        return word_vec

    def get_batch(self, batch, word_vec, emb_dim=300):
        # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        #         print(max_len)
        embed = np.zeros((max_len, len(batch), emb_dim))

        for i in range(len(batch)):
            for j in range(len(batch[i])):
                if batch[i][j] in word_vec:
                    embed[j, i, :] = word_vec[batch[i][j]]
                else:
                    embed[j, i, :] = word_vec['<oov>']
        #                     embed[j, i, :] = np.random.normal(size=(emb_dim))

        return torch.from_numpy(embed).float(), lengths

    def transform_text(self, data):
        # transform data into seq of embeddings
        premises = data['premises']
        hypotheses = data['hypotheses']

        # add bos and eos
        premises = [['<s>'] + premise + ['</s>'] for premise in premises]
        hypotheses = [['<s>'] + hypothese + ['</s>'] for hypothese in hypotheses]

        batches = []
        for stidx in range(0, len(premises), self.batch_size):
            # prepare batch
            s1_batch, s1_len = self.get_batch(premises[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            s2_batch, s2_len = self.get_batch(hypotheses[stidx:stidx + self.batch_size],
                                              self.word_vec, self.word_emb_dim)
            batches.append(((s1_batch, s1_len), (s2_batch, s2_len)))

        return batches


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


class NLIDataset_BERT(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 pretrained_dir,
                 max_seq_length=128,
                 batch_size=32):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, do_lower_case=True)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def convert_examples_to_features(self, examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        for (ex_index, (text_a, text_b)) in enumerate(examples):
            tokens_a = tokenizer.tokenize(' '.join(text_a))

            tokens_b = None
            if text_b:
                tokens_b = tokenizer.tokenize(' '.join(text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
        return features

    def transform_text(self, data):
        # transform data into seq of embeddings
        eval_features = self.convert_examples_to_features(list(zip(data['premises'], data['hypotheses'])),
                                                          self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)

        return eval_dataloader

# It calculates semantic similarity between two text inputs.
# text_ls (list): First text input either original text input or previous text.
# new_texts (list): Updated text inputs.
# idx (int): Index of the word that has been changed.
# sim_score_window (int): The number of words to consider around idx. If idx = -1 consider the whole text.
def calc_sim(text_ls, new_texts, idx, sim_score_window, sim_predictor):

    len_text = len(text_ls)
    half_sim_score_window = (sim_score_window - 1) // 2

    # Compute the starting and ending indices of the window.
    if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = idx - half_sim_score_window
        text_range_max = idx + half_sim_score_window + 1
    elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
        text_range_min = 0
        text_range_max = sim_score_window
    elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
        text_range_min = len_text - sim_score_window
        text_range_max = len_text
    else:
        text_range_min = 0
        text_range_max = len_text

    if text_range_min < 0:
        text_range_min = 0
    if text_range_max > len_text:
        text_range_max = len_text

    if idx == -1:
        text_rang_min = 0
        text_range_max = len_text

    semantic_sims = \
        sim_predictor.semantic_sim([' '.join(text_ls[text_range_min:text_range_max])],
            list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

    return semantic_sims

# Returns the hard label prediction of the target model.
# new_text (list): Text to be fed to target model.
# predictor: Target Model.
# orig_label (int): Original label.
# batch_size (int): Batch size.
def get_attack_result(hypotheses, premise, predictor, orig_label, batch_size):

    new_probs = predictor({'premises': [premise] * len(hypotheses), 'hypotheses': hypotheses})
    pr=(orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
    return pr

# It changes the inputt text at the specified index.
# rand_idx (int): Index to be mutated.
# text_ls (list): Original text.
# pos_ls (list): POS tage list.
# new_attack (list): The changed text during genetic optimization.
# best_attack (list): The best attack until now.
# remaining_indices (list): The indices in text input different from original input.
# synonyms_dict (dict): Synonym dict for each word.
# orig_label (int): Original prediction of the target model.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# batch_size (int): batch size.
def mutate(rand_idx, text_ls, pos_ls, premise, new_attack, best_attack, remaining_indices,
           synonyms_dict, old_syns, orig_label, sim_score_window,
           predictor, sim_predictor, batch_size):

    # Calculates the semantic similarity before mutation.
    random_text = new_attack[:]
    syns = synonyms_dict[text_ls[rand_idx]]
    prev_semantic_sims = calc_sim(text_ls, [best_attack], rand_idx, sim_score_window, sim_predictor)

    # Gives Priority to Original Word
    orig_word = 0
    if random_text[rand_idx] != text_ls[rand_idx]:

        temp_text = random_text[:]
        temp_text[rand_idx] = text_ls[rand_idx]
        pr = get_attack_result([temp_text], premise, predictor, orig_label, batch_size)
        semantic_sims = calc_sim(text_ls, [temp_text], rand_idx, sim_score_window, sim_predictor)
        if np.sum(pr) > 0:
            orig_word = 1
            return temp_text, 1 #(updated_text, queries_taken)

     # If replacing with original word does not yield adversarial text, then try to replace with other synonyms.
    if orig_word == 0:
        final_mask = []
        new_texts = []
        final_texts = []

         # Replace with synonyms.
        for syn in syns:

             # Ignore the synonym already present at position rand_idx.
            if syn == best_attack[rand_idx]:
                final_mask.append(0)
            else:
                final_mask.append(1)
            temp_text = random_text[:]
            temp_text[rand_idx] = syn
            new_texts.append(temp_text[:])

        # Filter out mutated texts that: (1) are not having same POS tag of the synonym, (2) lowers Semantic Similarity and (3) Do not satisfy adversarial criteria.
        synonyms_pos_ls = [criteria.get_pos(new_text[max(rand_idx - 4, 0):rand_idx + 5])[min(4, rand_idx)]
                               if len(new_text) > 10 else criteria.get_pos(new_text)[rand_idx] for new_text in new_texts]
        pos_mask = np.array(criteria.pos_filter(pos_ls[rand_idx], synonyms_pos_ls))
        semantic_sims = calc_sim(text_ls, new_texts, rand_idx, sim_score_window, sim_predictor)
        pr = get_attack_result(new_texts, premise, predictor, orig_label, batch_size)
        final_mask = np.asarray(final_mask)
        sem_filter = semantic_sims >= prev_semantic_sims[0]
        prediction_filter = pr > 0
        final_mask = final_mask*sem_filter
        final_mask = final_mask*prediction_filter
        final_mask = final_mask*pos_mask
        sem_vals = final_mask*semantic_sims

        for i in range(len(sem_vals)):
            if sem_vals[i] > 0:
                final_texts.append((new_texts[i], sem_vals[i]))

         # Return mutated text with best semantic similarity.
        final_texts.sort(key =  lambda x : x[1])
        final_texts.reverse()

        if len(final_texts) > 0:
            #old_syns[rand_idx].append(final_texts[0][0][rand_idx])
            return final_texts[0][0], len(new_texts)
        else:
            return [], len(new_texts)

# It generates children texts from the parent texts using crossover.
# population_size (int): Size of population used.
# population (list): The population currently in the optimization process.
# parent1_idx (int): The index of parent text input 1.
# parent2_idx (int): The index of parent text input 2.
# text_ls (list): Original text.
# best_attack (list): The best attack until now in the optimization.
# max_changes (int): The number of words substituted in the best_attack.
# changed_indices (list): The indices in text input different from original input.
# sim_score_window (int): The number of words to consider around idx.
# predictor: Target model.
# sim_predictor: USE to compute semantic similarity.
# orig_label (int): Original prediction of the target model.
# batch_size (int): batch size.
def crossover(population_size, population, parent1_idx, parent2_idx,
              text_ls, best_attack, max_changes, changed_indices,
              sim_score_window, sim_predictor,
              predictor, orig_label, batch_size):

    childs = []
    changes = []

    # Do crossover till population_size-1.
    for i in range(population_size-1):

        # Generates new child.
        p1 = population[parent1_idx[i]]
        p2 = population[parent2_idx[i]]
        assert len(p1) == len(p2)
        new_child = []
        for j in range(len(p1)):
            if np.random.uniform() < 0.5:
                new_child.append(p1[j])
            else:
                new_child.append(p2[j])
        change = 0
        cnt = 0
        mismatches = 0
        # Filter out crossover child which (1) Do not improve semantic similarity, (2) Have number of words substituted
        # more than the current best_attack.
        for k in range(len(changed_indices)):
            j = changed_indices[k]
            if new_child[j] == text_ls[j]:
                change+=1
                cnt+=1
            elif new_child[j] == best_attack[j]:
                change+=1
                cnt+=1
            elif new_child[j] != best_attack[j]:
                change+=1
                prev_semantic_sims = calc_sim(text_ls, [best_attack], j, sim_score_window, sim_predictor)
                semantic_sims = calc_sim(text_ls, [new_child], j, sim_score_window, sim_predictor)
                if semantic_sims[0] >= prev_semantic_sims[0]:
                    mismatches+=1
                    cnt+=1
        if cnt==change and mismatches<=max_changes:
            childs.append(new_child)
        changes.append(change)
    if len(childs) == 0:
        return [], 0

    # Filter out childs whoch do not satisfy the adversarial criteria.
    pr = get_attack_result(childs, predictor, orig_label, batch_size)
    final_childs = [childs[i] for i in range(len(pr)) if pr[i] > 0]
    return final_childs, len(final_childs)

def attack(fuzz_val, top_k_words, qrs, sample_index, hypotheses, premise, true_label,
           predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
           import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
           batch_size=32):

    # first check the prediction of the original text
    orig_probs = predictor({'premises': [premise], 'hypotheses': [hypotheses]}).squeeze() #predictor(premise,hypothese).squeeze()
    orig_label = torch.argmax(orig_probs)
    orig_prob = orig_probs.max()

    if true_label != orig_label:
        return '', 0, 0, orig_label, orig_label, 0, 0, 0
    else:
        text_ls = hypotheses[:]
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        if len_text < sim_score_window:
            sim_score_threshold = 0.1  # shut down the similarity thresholding function
        half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        # get the pos and verb tense info
        words_perturb = []
        pos_ls = criteria.get_pos(text_ls)
        pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        for pos in pos_pref:
            for i in range(len(pos_ls)):
                if pos_ls[i] == pos and len(text_ls[i]) > 2:
                    words_perturb.append((i, text_ls[i]))

        random.shuffle(words_perturb)

        # find synonyms and make a dict of synonyms of each word.
        words_perturb = words_perturb[:top_k_words]
        words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            synonym_values.append(temp)
        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        # STEP 1: Random initialisation.
        qrs = 0
        num_changed = 0
        flag = 0
        th = 0

        # Try substituting a random index with its random synonym.
        while qrs < len(text_ls):
            random_text = text_ls[:]
            for i in range(len(synonyms_all)):
                idx = synonyms_all[i][0]
                syn = synonyms_all[i][1]
                random_text[idx] = random.choice(syn)
                if i >= th:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs+=1
            th +=1
            if th > len_text:
                break
            if np.sum(pr)>0:
                flag = 1
                break
        old_qrs = qrs

         # If adversarial text is not yet generated try to substitute more words than 30%.
        while qrs < old_qrs + 2500 and flag == 0:
            random_text = text_ls[:]
            for j in range(len(synonyms_all)):
                idx = synonyms_all[j][0]
                syn = synonyms_all[j][1]
                random_text[idx] = random.choice(syn)
                if j >= len_text:
                    break
            pr = get_attack_result([random_text], premise, predictor, orig_label, batch_size)
            qrs+=1
            if np.sum(pr)>0:
                flag = 1
                break

        if flag == 1:
            #print("Found "+str(sample_index))
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1
            print(changed)

            # STEP 2: Search Space Reduction i.e.  Move Sample Close to Boundary
            while True:
                choices = []

                # For each word substituted in the original text, change it with its original word and compute
                # the change in semantic similarity.
                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        qrs+=1
                        pr = get_attack_result([new_text], premise, predictor, orig_label, batch_size)
                        if np.sum(pr) > 0:
                            choices.append((i,semantic_sims[0]))

                # Sort the relacements by semantic similarity and replace back the words with their original
                # counterparts till text remains adversarial.
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]
                        pr = get_attack_result([new_text], premise, predictor, orig_label, batch_size)
                        qrs+=1
                        if pr[0] == 0:
                            break
                        random_text[choices[i][0]] = text_ls[choices[i][0]]

                if len(choices) == 0:
                    break

            changed_indices = []
            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed_indices.append(i)
                    num_changed+=1
            print(str(num_changed)+" "+str(qrs))
            random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]
            #return '', 0, orig_label, orig_label, 0
            if num_changed == 1:
                return ' '.join(random_text), 1, 1, \
                    orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [random_text]})), qrs, random_sim, random_sim
            population_size = 30
            population = []
            old_syns = {}

            # STEP 3: Genetic Optimization
            # Genertaes initial population by mutating the substituted indices.
            for i in range(len(changed_indices)):
                txt, mut_qrs = mutate(changed_indices[i], text_ls, pos_ls, premise, random_text, random_text, changed_indices,
                                synonyms_dict, old_syns, orig_label, sim_score_window,
                                predictor, sim_predictor, batch_size)
                qrs+=mut_qrs
                if len(txt)!=0:
                    population.append(txt)
            max_iters = 1000
            pop_count = 0
            attack_same = 0
            old_best_attack = random_text[:]
            if len(population) == 0:
                return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                            orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [random_text]})), qrs, random_sim, random_sim

            ## Genetic Optimization
            for _ in range(max_iters):
                max_changes = len_text

                # Find the best_attack text in the current population.
                for txt in population:
                    changes = 0
                    for i in range(len(changed_indices)):
                        j = changed_indices[i]
                        if txt[j] != text_ls[j]:
                            changes+=1
                    if changes <= max_changes:
                        max_changes = changes
                        best_attack = txt

                # Check that it is adversarial.
                pr = get_attack_result([best_attack], premise, predictor, orig_label, batch_size)
                assert pr[0] > 0
                flag = 0

                # If the new best attack is the same as the old best attack for last 15 consecutive iterations tham
                # stop optimization.
                for i in range(len(changed_indices)):
                    k = changed_indices[i]
                    if best_attack[k] != old_best_attack[k]:
                        flag = 1
                        break
                if flag == 1:
                    attack_same = 0
                else:
                    attack_same+=1

                if attack_same >= 15:
                    sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                    return ' '.join(best_attack), max_changes, len(changed_indices), \
                         orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [best_attack]})), qrs, sim, random_sim

                old_best_attack = best_attack[:]

                print(str(max_changes)+" After Genetic")

                # If only 1 input word substituted return it.
                if max_changes == 1:
                    sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]
                    return ' '.join(best_attack), max_changes, len(changed_indices), \
                         orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [best_attack]})), qrs, sim, random_sim

                 # Sample two parent input propotional to semantic similarity.
                sem_scores = calc_sim(text_ls, population, -1, sim_score_window, sim_predictor)
                sem_scores = np.asarray(sem_scores)
                scrs = softmax(sem_scores)
                parent1_idx = np.random.choice(len(population), size = population_size-1, p = scrs)
                parent2_idx = np.random.choice(len(population), size = population_size-1, p = scrs)
 
                ## Crossover
                final_childs, cross_qrs = crossover(population_size, population, parent1_idx, parent2_idx,
                                         text_ls, premise, best_attack, max_changes, changed_indices, sim_score_window, sim_predictor,
                                         predictor, orig_label, batch_size)
                qrs+=cross_qrs
                population = []
                indices_done = []

                # Randomly select indices for mutation from the changed indices. The changed indices contains indices
                # which has not been replaced by original word.
                indices = np.random.choice(len(changed_indices), size = min(len(changed_indices), len(final_childs)))
                for i in range(len(indices)):
                    child = final_childs[i]
                    j = indices[i]
                     # If the index has been substituted no need to mutate.
                    if text_ls[changed_indices[j]] == child[changed_indices[j]]:
                        population.append(child)
                        indices_done.append(j)
                        continue

                    # Mutate the childs obtained after crossover on the random index.
                    txt, mut_qrs = mutate(changed_indices[j], text_ls, pos_ls, premise, child, child, changed_indices,
                                            synonyms_dict, old_syns, orig_label, sim_score_window,
                                            predictor, sim_predictor, batch_size)
                    qrs+=mut_qrs
                    indices_done.append(j)

                    # If the input has been mutated successfully add to population for nest generation.
                    if len(txt)!=0:
                        population.append(txt)
                if len(population) == 0:
                    pop_count+=1
                else:
                    pop_count = 0

                 # If length of population is zero for 15 consecutive iterations return.
                if pop_count >= 15:
                    sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

                    return ' '.join(best_attack), len(changed_indices), \
                         max_changes, orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [best_attack]})), qrs, sim, random_sim

                # Add best adversarial attack text also to next population.
                population.append(best_attack)
            sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)[0]

            return ' '.join(best_attack), max_changes, len(changed_indices), \
                  orig_label, torch.argmax(predictor({'premises':[premise], 'hypotheses': [best_attack]})), qrs, sim, random_sim

        else:
            print("Not Found")
            return '', 0,0, orig_label, orig_label, 0, 0, 0


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="Which dataset to attack.")
    parser.add_argument("--target_model",
                        type=str,
                        required=True,
                        choices=['infersent', 'esim', 'bert'],
                        help="Target models for text classification: fasttext, charcnn, word level lstm "
                             "For NLI: InferSent, ESIM, bert-base-uncased")
    parser.add_argument("--output_dir",
                        type=str,
                        default='adv_results',
                        help="The output directory where the attack results will be written.")
    parser.add_argument("--target_model_path",
                        type=str,
                        required=True,
                        help="pre-trained target model path")
    parser.add_argument("--word_embeddings_path",
                        type=str,
                        default='',
                        help="path to the word embeddings for the target model")
    parser.add_argument("--counter_fitting_embeddings_path",
                        type=str,
                        default="counter-fitted-vectors.txt",
                        help="path to the counter-fitting embeddings we used to find synonyms")
    parser.add_argument("--counter_fitting_cos_sim_path",
                        type=str,
                        default='',
                        help="pre-compute the cosine similarity scores based on the counter-fitting embeddings")
    parser.add_argument("--USE_cache_path",
                        type=str,
                        required=True,
                        help="Path to the USE encoder cache.")

    ## Model hyperparameters
    parser.add_argument("--sim_score_window",
                        default=310,
                        type=int,
                        help="Text length or token number to compute the semantic similarity score")
    parser.add_argument("--import_score_threshold",
                        default=-1.,
                        type=float,
                        help="Required mininum importance score.")
    parser.add_argument("--sim_score_threshold",
                        default=0.47,
                        type=float,
                        help="Required minimum semantic similarity score.")
    parser.add_argument("--synonym_num",
                        default=50,
                        type=int,
                        help="Number of synonyms to extract")
    parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        help="Batch size to get prediction")
    parser.add_argument("--data_size",
                        default=1000,
                        type=int,
                        help="Data size to create adversaries")
    parser.add_argument("--perturb_ratio",
                        default=0.,
                        type=float,
                        help="Whether use random perturbation for ablation study")
    parser.add_argument("--target_dataset",
                        default="imdb",
                        type=str,
                        help="Dataset Name")
    parser.add_argument("--fuzz",
                        default=0,
                        type=int,
                        help="Word Pruning Value")
    parser.add_argument("--top_k_words",
                        default=1000000,
                        type=int,
                        help="Top K Words")
    parser.add_argument("--allowed_qrs",
                        default=1000000,
                        type=int,
                        help="Allowerd qrs")

    args = parser.parse_args()
    log_file = "results_nli_hard_label/"+args.target_model+"/"+args.target_dataset+"/log.txt"
    result_file = "results_nli_hard_label/"+args.target_model+"/"+args.target_dataset+"/results_final.csv"
    Path(result_file).mkdir(parents=True, exist_ok=True)
    Path(log_file).mkdir(parents=True, exist_ok=True)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    else:
        os.makedirs(args.output_dir, exist_ok=True)

    # get data to attack, fetch first [args.data_size] data samples for adversarial attacking
    data = read_data(args.dataset_path, data_size=args.data_size, target_model=args.target_model)
    print("Data import finished!")

    # construct the model
    print("Building Model...")
    if args.target_model == 'esim':
        model = NLI_infer_ESIM(args.target_model_path,
                                args.word_embeddings_path,
                               batch_size=args.batch_size)
    elif args.target_model == 'infersent':
        model = NLI_infer_InferSent(args.target_model_path,
                                    args.word_embeddings_path,
                                    data=data,
                                    batch_size=args.batch_size)
    else:
        model = NLI_infer_BERT(args.target_model_path)
    predictor = model.text_pred
    print("Model built!")

    # prepare synonym extractor
    # build dictionary via the embedding file
    print("Building vocab...")
    idx2word = {}
    word2idx = {}
    sim_lis=[]
    with open(args.counter_fitting_embeddings_path, 'r') as ifile:
        for line in ifile:
            word = line.split()[0]
            if word not in idx2word:
                idx2word[len(idx2word)] = word
                word2idx[word] = len(idx2word) - 1

    # for cosine similarity matrix
    print("Building cos sim matrix...")
    if args.counter_fitting_cos_sim_path:
        # load pre-computed cosine similarity matrix if provided
        print('Load pre-computed cosine similarity matrix from {}'.format(args.counter_fitting_cos_sim_path))
        with open(args.counter_fitting_cos_sim_path, "rb") as fp:
            sim_lis = pickle.load(fp)
    else:
        # calculate the cosine similarity matrix
        print('Start computing the cosine similarity matrix!')
        embeddings = []
        with open(args.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                embedding = [float(num) for num in line.strip().split()[1:]]
                embeddings.append(embedding)
        embeddings = np.array(embeddings)
        print(embeddings.T.shape)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = np.asarray(embeddings / norm, "float64")
        cos_sim = np.dot(embeddings, embeddings.T)
    print("Cos sim import finished!")

    # build the semantic similarity module
    use = USE(args.USE_cache_path)

    # start attacking
    orig_failures = 0.
    adv_failures = 0.
    avg=0.
    tot = 0
    changed_rates = []
    nums_queries = []
    orig_texts = []
    adv_texts = []
    true_labels = []
    new_labels = []
    wrds=[]
    s_queries=[]
    f_queries=[]
    success=[]
    results=[]
    fails=[]
    final_sims = []
    random_sims = []
    random_changed_rates = []
    adv_rows = []
    stop_words_set = criteria.get_stopwords()

    for idx, premise in enumerate(data['premises']):

        if idx % 100 == 0:
            print(np.mean(changed_rates))
            #print(len(success))
            print('{} samples out of {} have been finished!'.format(idx, args.data_size))

        hypothese, true_label = data['hypotheses'][idx], data['labels'][idx]
        #print(hypothese)
        if args.perturb_ratio == 0:
            new_text, num_changed, random_changed, orig_label, \
            new_label, num_queries, sim, random_sim = attack(args.fuzz,args.top_k_words,args.allowed_qrs,
                                            idx, hypothese, premise, true_label, predictor, stop_words_set,
                                            word2idx, idx2word, sim_lis , sim_predictor=use,
                                            sim_score_threshold=args.sim_score_threshold,
                                            import_score_threshold=args.import_score_threshold,
                                            sim_score_window=args.sim_score_window,
                                            synonym_num=args.synonym_num,
                                            batch_size=args.batch_size)
        if true_label != orig_label:
            orig_failures += 1
        else:
            nums_queries.append(num_queries)
        if true_label != new_label:
            adv_failures += 1

        changed_rate = 1.0 * num_changed / len(hypothese)
        random_changed_rate = 1.0 * random_changed / len(hypothese)
        # print('orig sentence ({}):'.format(orig_label), ' '.join(text), '\nto new sentence ({}):'.format(new_label),
        #       new_text, '\n{}/{} changed at {:.2f}%'.format(num_changed, len(text), changed_rate * 100))
        if true_label == orig_label and true_label != new_label:
            temp=[]
            s_queries.append(num_queries)
            success.append(idx)
            changed_rates.append(changed_rate)
            orig_texts.append(' '.join(hypothese))
            adv_texts.append(new_text)
            true_labels.append(true_label)
            new_labels.append(new_label)
            random_changed_rates.append(random_changed_rate)
            random_sims.append(random_sim)
            final_sims.append(sim)
            temp.append(idx)
            temp.append(orig_label)
            temp.append(new_label)
            temp.append(' '.join(hypothese))
            temp.append(new_text)
            temp.append(num_queries)
            temp.append(random_sim)
            temp.append(sim)
            temp.append(changed_rate * 100)
            temp.append(random_changed_rate * 100)
            results.append(temp)
            print("Attacked: "+str(idx))

    with open(result_file, 'w') as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
      
    # writing the data rows 
        csvwriter.writerows(results)



    message = 'For target model using TFIDF {} on dataset window size {} with WP val {} top words {} qrs {} : ' \
              'original accuracy: {:.3f}%, adv accuracy: {:.3f}%, random avg  change: {:.3f}% ' \
              'avg changed rate: {:.3f}%, num of queries: {:.1f}, random_sims: {:.3f}%, final_sims : {:.3f}% \n'.format(args.target_model,
                                                                      args.sim_score_window,
                                                                      args.fuzz,
                                                                      args.top_k_words,args.allowed_qrs,
                                                                     (1-orig_failures/1000)*100,
                                                                     (1-adv_failures/1000)*100,
                                                                     np.mean(random_changed_rates)*100,
                                                                     np.mean(changed_rates)*100,
                                                                     np.mean(nums_queries),
                                                                     np.mean(random_sims),
                                                                     np.mean(final_sims))

    print(message)
    print(orig_failures)

    log=open(log_file,'a')
    log.write(message)
    with open(result_file,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)

    if args.target_model == 'bert':
        labeldict = {0: "contradiction",
                     1: "entailment",
                     2:  "neutral"}
    else:
        labeldict = {0: "entailment",
                     1: "neutral",
                     2: "contradiction"}

    with open(os.path.join(args.output_dir, 'adversaries.txt'), 'w') as ofile:
        for orig_premise, orig_hypothesis, adv_hypothesis, \
            true_label, new_label in zip(orig_premises, orig_hypotheses, adv_hypotheses,
                                        true_labels, new_labels):
            ofile.write('orig premise:\t{}\norig hypothesis ({}):\t{}\n'
                        'adv hypothesis ({}):\t{}\n\n'.format(orig_premise,
                                                              labeldict[true_label],
                                                              orig_hypothesis,
                                                              labeldict[new_label],
                                                              adv_hypothesis))

if __name__ == "__main__":
    main()
