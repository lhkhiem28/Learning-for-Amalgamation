import time
import json
import tqdm
import warnings
import numpy as np
import pandas as pd
import sklearn.model_selection as model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils as utils
from transformers import BertConfig
from transformers import BertTokenizer
from transformers import BertPreTrainedModel
from BERT.models.bert import BertFC
from BERT.models.bertrnn import BertLSTM, BertGRU
from BERT.models.bertcnn import BertCNN
from BERT.models.bertrcnn import BertLSTMCNN, BertGRUCNN
from BERT.models.bertrnnattn import BertLSTMAttn, BertGRUAttn
from BERT.models.berttransformer import BertTransformer
from BERT.models.MoE import MoE
from BERT.data import ReviewDataset
import sklearn


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')


def onehot_labels(labels):
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoded = label_encoder.fit_transform(labels)
    label_encoded = label_encoded.reshape(len(label_encoded), 1)

    onehot_encoder = sklearn.preprocessing.OneHotEncoder(sparse=False)
    onehot_encoded = onehot_encoder.fit_transform(label_encoded)

    return onehot_encoded

def evaluate(model, criterion, dataset, batch_size):
    dataloader = utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    #for inputs, labels in tqdm.tqdm(dataloader):
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        loss, outputs = model(input_ids=inputs, labels=labels)
        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(dataset)
    total_acc = running_corrects.double() / len(dataset)
    precision = precision_score(total_labels, total_preds)
    recall = recall_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds)
    auc = roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


def load_word_embedding(embedding_file):
    #print('Loading word embeddings...')
    embeddings_index = {}
    f = codecs.open(embedding_file, encoding='utf-8')
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def filter_glove_emb(word_dict, embedding_index):
    # filter embeddings
    dim = 300
    scale = np.sqrt(3.0 / dim)
    #vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])
    vectors = np.random.uniform(-scale, scale, [len(word_dict), dim])

    for word in word_dict.keys():
        embedding_vector = embedding_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            index = list(word_dict.keys()).index(word)
            vectors[index] = embedding_vector
        #else:
        #    print(word)
    return vectors



import re
import pickle

import string
import codecs
import pandas as pd
import numpy as np
from torchtext import data
from torchtext.data import Dataset, Example
from sklearn.model_selection import train_test_split
#from utils.utils import str2list
from pyvi import ViTokenizer
import torch

class SeriesExample(Example):
    """Class to convert a pandas Series to an Example"""

    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)

    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                                 "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex

class DataFrameDataset(Dataset):
    def __init__(self, examples, fields, filter_pred=None):
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

def preprocessing(text, embeddings_index):
    text_ori = text

    # remove duplicate characters such as đẹppppppp
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)

    # remove punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)

    # remove '_'
    text = text.replace('_', ' ')

    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()])

    # lower word
    text = text.lower()

    # replace special words
    replace_list = {
        'ô kêi': ' ok ', 'o kê': ' ok ',
        'kh ':' không ', 'kô ':' không ', 'hok ':' không ',
        'kp ': ' không phải ', 'kô ': ' không ', 'ko ': ' không ', 'khong ': ' không ', 'hok ': ' không ',
    }
    for k, v in replace_list.items():
        text = text.replace(k, v)

    # split texts
    texts = text.split()
    texts = [t for t in texts if embeddings_index.get(t) is not None]
    text = u' '.join(texts)

    if len(texts) < 5:
        text = None

    return text

def create_vocab(dataset, nrows):
    embeddings_index = load_word_embedding('/data/cuong/data/nlp/embedding/cc.vi.300.vec')

    text_column = 'discriptions'
    label_column = 'mapped_rating'
    use_cols = [text_column, label_column]
    df_train = pd.read_csv('BERT/dataset/' + dataset + '/train.csv', usecols=use_cols, nrows=nrows)
    df_train.dropna(subset=[text_column], inplace=True)
    df_train[text_column] = df_train[text_column].apply(lambda x:preprocessing(x, embeddings_index))

    df_train.drop_duplicates(subset=text_column, keep = 'first', inplace = True)
    df_train.dropna(subset=[text_column], inplace=True)
    df_train = df_train.reset_index()

    TEXT = data.Field(sequential=True, lower=True, include_lengths=True, fix_length=None)
    fields = {}
    fields[text_column] = TEXT
    train_dataset = DataFrameDataset(df_train, fields=fields)

    TEXT.build_vocab(train_dataset)
    vectors = filter_glove_emb(TEXT.vocab.stoi, embeddings_index)
    TEXT.vocab.set_vectors(TEXT.vocab.stoi, torch.from_numpy(vectors), 300)

    del df_train
    del embeddings_index

    return TEXT.vocab


def train(head_model, dataset, num_epochs, batch_size, learning_rate, nrows=None):
    train = pd.read_csv('BERT/dataset/' + dataset + '/train.csv', nrows=nrows)
    print('train dataset', train.shape)
    print('train={}; counts={}'.format(train.shape, train['mapped_rating'].value_counts()))

    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    train_dataset = ReviewDataset(list(train['discriptions']), train['mapped_rating'].values, tokenizer)
    valid_dataset = ReviewDataset(list(valid['discriptions']), valid['mapped_rating'].values, tokenizer)

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    config = BertConfig.from_pretrained('bert-base-multilingual-cased')
    config.num_labels = 2

    if head_model == 'fc':
        model = BertFC(config=config, dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)

    elif head_model == 'lstm':
        model = BertLSTM(config=config, hidden_size=128, dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)
    elif head_model == 'gru':
        model = BertGRU(config=config, hidden_size=128, dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)
    elif head_model == 'lstm-attn':
        model = BertLSTMAttn(config=config, hidden_size=128, dropout_prob=0.1, attention_type='general')
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)

    elif head_model == 'gru-attn':
        model = BertGRUAttn(config=config, hidden_size=128, dropout_prob=0.1, attention_type='general')
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)
    elif head_model == 'lstm-cnn':
        model = BertLSTMCNN(config=config, hidden_size=128, dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        
    elif head_model == 'gru-cnn':
        model = BertGRUCNN(config=config, hidden_size=128, dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)
    elif head_model == 'cnn':
        model = BertCNN(config=config, n_filters=128, kernel_sizes=[1, 3, 5], dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)
    elif head_model == 'transformer':
        model = BertTransformer(config=config, num_layers=2, num_heads=8, maxlen=128, dropout_prob=0.1)
        model = model.from_pretrained('bert-base-multilingual-cased', config=config)
    elif head_model == 'moe':
        # load fastext embedding
        vocab = create_vocab(dataset, nrows)
        print('vocab', len(vocab.vectors))
        print(vocab.vectors)


        # model
        model = MoE(config=config, vocab=vocab, dropout_prob=0.1)
        model.bert_emb = model.bert_emb.from_pretrained('bert-base-multilingual-cased', config=config)

    model = model.to(device)
    print(model)

    print('head_model', head_model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    since = time.time()
    history = {
        'train': {'loss': [], 'acc': []},
        'valid': {'loss': [], 'acc': []},
        'lr': []
    }
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        head = 'epoch {:2}/{:2}'.format(epoch, num_epochs)
        print(head + '\n' + '-'*(len(head)))

        model.train()
        running_loss = 0.0
        running_corrects = 0
        #for inputs, labels in tqdm.tqdm(train_loader):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            #model(input_ids=inputs, labels=labels)
            loss, outputs = model(input_ids=inputs, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        history['train']['loss'].append(epoch_loss)
        history['train']['acc'].append(epoch_acc.item())
        print('{} - loss: {:.4f} acc: {:.2f}'.format('train', epoch_loss, epoch_acc * 100))

        model.eval()
        running_loss = 0.0
        running_corrects = 0
        #for inputs, labels in tqdm.tqdm(valid_loader):
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            loss, outputs = model(input_ids=inputs, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            running_loss += error.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        history['valid']['loss'].append(epoch_loss)
        history['valid']['acc'].append(epoch_acc.item())
        print('{} - loss: {:.4f} acc: {:.2f}'.format('valid', epoch_loss, epoch_acc * 100))

        history['lr'].append(optimizer.param_groups[0]['lr'])

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'BERT/logs/' + dataset + '/bert_{}.pth'.format(head_model))

        cnt += 1
        if cnt > early_stopping:
            break

    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing


    test = pd.read_csv('BERT/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDataset(list(test['discriptions']), test['mapped_rating'].values, tokenizer)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('BERT/logs/' + dataset + '/bert_{}.pth'.format(head_model)))
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    #for inputs, labels in tqdm.tqdm(dataloader):
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        loss, outputs = model(input_ids=inputs, labels=labels)
        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())


    print('dataset', len(dataset))
    print('test_dataset', len(test_dataset))


    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))

    #evaluate(model, criterion=nn.CrossEntropyLoss(), dataset=test_dataset, batch_size=batch_size)
