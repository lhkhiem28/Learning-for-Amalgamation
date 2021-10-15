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

from transformers import XLMConfig
from transformers import XLMTokenizer

from main.models.bert import *
from main.models.phobert import *
from main.models.xlm import *

from main.models.moe import *

from main.data import *
import sklearn
from transformers import RobertaConfig, BertConfig, BertTokenizer
from fairseq.data import Dictionary
# from vncorenlp import VnCoreNLP


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings('ignore')

import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Saigon')



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


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_bert(dataset, num_epochs, batch_size, learning_rate):
    print('train_bert ... ')

    # prepare data
    train = pd.read_csv('main/dataset/' + dataset + '/train.csv')
    
    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    train_dataset = ReviewDatasetBert(list(train['discriptions']), train['mapped_rating'].values, tokenizer)
    valid_dataset = ReviewDatasetBert(list(valid['discriptions']), valid['mapped_rating'].values, tokenizer)

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # load model
    config_bert = BertConfig.from_pretrained('bert-base-multilingual-cased')
    config_bert.num_labels = 2

    model = Bert(config_bert, hidden_size=128, dropout_prob=0.1)
    model.bert = model.bert.from_pretrained('bert-base-multilingual-cased', config=config_bert)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)

    print('train dataset', train.shape)
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    since = time.time()
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==========================================================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        print('\n')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs_bert, labels in (train_loader):
            inputs_bert, labels = inputs_bert.to(device), labels.to(device)

            optimizer.zero_grad()

            loss, outputs, _ = model(input_ids_bert=inputs_bert, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs_bert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('[TRAIN] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs_bert, labels in (valid_loader):
            inputs_bert, labels = inputs_bert.to(device), labels.to(device)

            loss, outputs, _ = model(input_ids_bert=inputs_bert, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            running_loss += error.item() * inputs_bert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        print('[VAL] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'main/logs/' + dataset + '/bert.pth')

        cnt += 1
        if cnt > early_stopping:
            break


    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    print("Start testing ...\n" + "==========================================================\n")

    test = pd.read_csv('main/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDatasetBert(list(test['discriptions']), test['mapped_rating'].values, tokenizer)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('main/logs/' + dataset + '/bert.pth'))

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs_bert, labels in (test_loader):
        inputs_bert, labels = inputs_bert.to(device), labels.to(device)

        loss, outputs, _ = model(input_ids_bert=inputs_bert, labels=labels)
        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs_bert.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


def train_phobert(dataset, num_epochs, batch_size, learning_rate):
    print('train_phobert ... ')

    # prepare data
    train = pd.read_csv('main/dataset/' + dataset + '/train.csv')
    
    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    vocab = Dictionary()
    vocab.add_from_file('main/phobert/dict.txt')
    segmenter = VnCoreNLP('main/vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx500m')

    train_dataset = ReviewDatasetPhoBert(list(train['discriptions']), train['mapped_rating'].values, segmenter, vocab)
    valid_dataset = ReviewDatasetPhoBert(list(valid['discriptions']), valid['mapped_rating'].values, segmenter, vocab)

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # load model
    config_phobert = RobertaConfig.from_pretrained('main/phobert/config.json')
    config_phobert.num_labels = 2

    model = PhoBert(config_phobert, hidden_size=128, dropout_prob=0.1)
    model.phobert = model.phobert.from_pretrained('main/phobert/model.bin', config=config_phobert)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('phobert pytorch_total_params', pytorch_total_params)

    print('train dataset', train.shape)
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    since = time.time()
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==========================================================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        print('\n')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs_phobert, labels in (train_loader):
            inputs_phobert, labels = inputs_phobert.to(device), labels.to(device)

            optimizer.zero_grad()

            loss, outputs, _ = model(input_ids_phobert=inputs_phobert, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs_phobert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('[TRAIN] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs_phobert, labels in (valid_loader):
            inputs_phobert, labels = inputs_phobert.to(device), labels.to(device)

            loss, outputs, _ = model(input_ids_phobert=inputs_phobert, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            running_loss += error.item() * inputs_phobert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        print('[VAL] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'main/logs/' + dataset + '/phobert.pth')

        cnt += 1
        if cnt > early_stopping:
            break

    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    print("Start testing ...\n" + "==========================================================\n")

    test = pd.read_csv('main/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDatasetPhoBert(list(test['discriptions']), test['mapped_rating'].values, segmenter, vocab)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('main/logs/' + dataset + '/phobert.pth'))

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs_phobert, labels in (test_loader):
        inputs_phobert, labels = inputs_phobert.to(device), labels.to(device)

        loss, outputs, _ = model(input_ids_phobert=inputs_phobert, labels=labels)
        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs_phobert.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


def train_xlm(dataset, num_epochs, batch_size, learning_rate):
    print('Traing XLM ...')

    # prepare data
    train = pd.read_csv('main/dataset/' + dataset + '/train.csv')
    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024')
    #tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-tlm-xnli15-1024') #bad

    train_dataset = ReviewDatasetXLM(list(train['discriptions']), train['mapped_rating'].values, tokenizer)
    valid_dataset = ReviewDatasetXLM(list(valid['discriptions']), valid['mapped_rating'].values, tokenizer)

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # load model
    config_xlm = XLMConfig.from_pretrained('xlm-mlm-xnli15-1024')
    #config_xlm = XLMConfig.from_pretrained('xlm-mlm-tlm-xnli15-1024')
    config_xlm.num_labels = 2

    model = Xlm(config_xlm, hidden_size=128, dropout_prob=0.1)
    model.xlm = model.xlm.from_pretrained('xlm-mlm-xnli15-1024', config=config_xlm)
    #model.xlm = model.xlm.from_pretrained('xlm-mlm-tlm-xnli15-1024', config=config_xlm)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)

    print('train dataset', train.shape)
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    since = time.time()
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==========================================================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        print('\n')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs_xlm, labels in (train_loader):
            inputs_xlm, labels = inputs_xlm.to(device), labels.to(device)

            optimizer.zero_grad()
            loss, outputs, _ = model(input_ids_xlm=inputs_xlm, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs_xlm.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('[TRAIN] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs_xlm, labels in (valid_loader):
            inputs_xlm, labels = inputs_xlm.to(device), labels.to(device)

            loss, outputs, _ = model(input_ids_xlm=inputs_xlm, labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            running_loss += error.item() * inputs_xlm.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        print('[VAL] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'main/logs/' + dataset + '/xlm.pth')

        cnt += 1
        if cnt > early_stopping:
            break

    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    print("Start testing ...\n" + "==========================================================\n")

    test = pd.read_csv('main/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDatasetXLM(list(test['discriptions']), test['mapped_rating'].values, tokenizer)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('main/logs/' + dataset + '/xlm.pth'))

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs_xlm, labels in (test_loader):
        inputs_xlm, labels = inputs_xlm.to(device), labels.to(device)

        loss, outputs, _ = model(input_ids_xlm=inputs_xlm, labels=labels)
        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs_xlm.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


class LSTMCNN(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=128, dropout_prob=0.2):
        super(LSTMCNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_matrix.shape[1], self.hidden_size, bidirectional=True)
        self.drop = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(embedding_matrix.shape[1] + 2*self.hidden_size, self.hidden_size)
        self.extracter  = nn.Linear(self.hidden_size, 256)
        self.classifier = nn.Linear(256, 2)

    def forward(self, x, labels=None):
        embedding = self.embedding(x).permute(1, 0, 2)
        lstm = self.lstm(embedding)[0]

        cat = torch.cat((lstm[:, :, :self.hidden_size], embedding, lstm[:, :, self.hidden_size:]), 2).permute(1, 0, 2)
        cat = torch.tanh(self.linear(cat)).permute(0, 2, 1)
        cat = F.max_pool1d(cat, cat.shape[2]).squeeze(2)
        fea = self.extracter(cat)
        out = self.drop(fea)
        out = self.classifier(out)

        if labels is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out.view(-1, 2), labels.view(-1))
            return loss, fea, out

        return fea, out

from keras.preprocessing.text import Tokenizer
from torch.utils.data import TensorDataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
import torch.optim.lr_scheduler as lr_scheduler

def load_tokenizer(df_path, num_words=100000):
    train_df, test_df = pd.read_csv(df_path + "train.csv"), pd.read_csv(df_path + "test.csv")

    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(list(train_df["discriptions"].astype(str).values) + list(test_df["discriptions"].astype(str).values))

    return tokenizer

def load_embedding(embedding_path, embedding_size, word_index):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype="float32")

    embedding_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_path, encoding="utf-8", errors="ignore"))

    all_embeds = np.stack(embedding_index.values())
    embed_mean, embed_std = all_embeds.mean(), all_embeds.std()
    embedding_matrix = np.random.normal(embed_mean, embed_std, (len(word_index) + 1, embedding_size))

    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def train_lstmcnn(dataset, num_epochs, batch_size, learning_rate):
    print('Training LSTM CNN ...')

    # prepare data
    train = pd.read_csv('main/dataset/' + dataset + '/train.csv')
    
    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    tokenizer = load_tokenizer('main/dataset/' + dataset + '/')
    embedding_matrix = load_embedding("main/embedding/cc.en.300.vec", embedding_size=300, word_index=tokenizer.word_index)

    train_tokenized = tokenizer.texts_to_sequences(train["discriptions"].astype(str))
    X_train, y_train = pad_sequences(train_tokenized, maxlen=100), train["mapped_rating"].values
    X_train, y_train = torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_tokenized = tokenizer.texts_to_sequences(valid["discriptions"].astype(str))
    X_valid, y_valid = pad_sequences(valid_tokenized, maxlen=100), valid["mapped_rating"].values
    X_valid, y_valid = torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.long)
    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    # model
    model = LSTMCNN(embedding_matrix)
    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)

    print('train dataset', train.shape)
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))

    #optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, verbose=True)
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()

    since = time.time()
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==========================================================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        print('\n')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in (train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            loss, fea, out = model(inputs, labels=labels)
            preds = torch.max(out, 1)[1]
            error = criterion(out, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('[TRAIN] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in (valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            loss, fea, out = model(inputs, labels=labels)
            preds = torch.max(out, 1)[1]
            error = criterion(out, labels)

            running_loss += error.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        print('[VAL] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'main/logs/' + dataset + '/lstmcnn.pth')

        cnt += 1
        if cnt > early_stopping:
            break

    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    print("Start testing ...\n" + "==========================================================\n")

    test = pd.read_csv('main/dataset/' + dataset + '/test.csv')
    test_tokenized = tokenizer.texts_to_sequences(test["discriptions"].astype(str))
    X_test, y_test = pad_sequences(test_tokenized, maxlen=100), test["mapped_rating"].values
    X_test, y_test = torch.tensor(X_test, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('main/logs/' + dataset + '/lstmcnn.pth'))

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs, labels in (test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        loss, fea, out = model(inputs, labels=labels)
        scores = F.softmax(out, dim=1)
        preds = torch.max(out, 1)[1]
        error = criterion(out, labels)

        running_loss += error.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


def train_moe(dataset, num_epochs, batch_size, learning_rate):
    print('Training MOE of PhoBert, Bert and XLM.')

    # prepare data
    train = pd.read_csv('main/dataset/' + dataset + '/train.csv')
    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    # for bert data
    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tokenizer_xlm  = XLMTokenizer.from_pretrained('xlm-mlm-xnli15-1024')

    # for phobert data
    vocab = Dictionary()
    vocab.add_from_file('main/phobert/dict.txt')
    segmenter = VnCoreNLP('main/vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx500m')

    train_dataset = ReviewDataset(list(train['discriptions']), train['mapped_rating'].values,
                                    segmenter, vocab, #phobert
                                    tokenizer_bert, #bert
                                    tokenizer_xlm #xlm
                                    )
    valid_dataset = ReviewDataset(list(valid['discriptions']), valid['mapped_rating'].values,
                                    segmenter, vocab, #phobert
                                    tokenizer_bert, #bert
                                    tokenizer_xlm #xlm
                                    )

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # load model
    config_phobert = RobertaConfig.from_pretrained('main/phobert/config.json')
    config_bert = BertConfig.from_pretrained('bert-base-multilingual-cased')
    config_xlm = XLMConfig.from_pretrained('xlm-mlm-xnli15-1024')
    config_phobert.num_labels = 2
    config_bert.num_labels = 2
    config_xlm.num_labels = 2

    model = Moe(config_phobert, config_bert, config_xlm, hidden_size=128, dropout_prob=0.1)

    model.phobert.load_state_dict(torch.load('main/logs/' + dataset + '/phobert.pth'))
    model.bert.load_state_dict(torch.load('main/logs/' + dataset + '/bert.pth'))
    model.xlm.load_state_dict(torch.load('main/logs/' + dataset + '/xlm.pth'))

    for param in model.phobert.parameters():
        param.requires_grad = False
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.xlm.parameters():
        param.requires_grad = False

    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)

    print('train dataset', train.shape)
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    since = time.time()
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==========================================================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        print('\n')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs_phobert, inputs_bert, inputs_xlm, labels in (train_loader):
            inputs_phobert = inputs_phobert.to(device)
            inputs_bert = inputs_bert.to(device)
            inputs_xlm = inputs_xlm.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            loss, outputs = model(  input_ids_phobert=inputs_phobert,
                                    input_ids_bert=inputs_bert,
                                    input_ids_xlm=inputs_xlm,
                                    labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs_phobert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('[TRAIN] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs_phobert, inputs_bert, inputs_xlm, labels in (valid_loader):
            inputs_phobert = inputs_phobert.to(device)
            inputs_bert = inputs_bert.to(device)
            inputs_xlm = inputs_xlm.to(device)
            labels = labels.to(device)

            loss, outputs = model(  input_ids_phobert=inputs_phobert,
                                    input_ids_bert=inputs_bert,
                                    input_ids_xlm=inputs_xlm,
                                    labels=labels)

            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            running_loss += error.item() * inputs_phobert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        print('[VAL] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'main/logs/' + dataset + '/moe.pth')

        cnt += 1
        if cnt > early_stopping:
            break

    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    print("Start testing ...\n" + "==========================================================\n")

    test = pd.read_csv('main/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDataset(list(test['discriptions']), test['mapped_rating'].values,
                                segmenter, vocab, #phobert
                                tokenizer_bert, #bert
                                tokenizer_xlm, #xlm
                                )
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('main/logs/' + dataset + '/moe.pth'))

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs_phobert, inputs_bert, inputs_xlm, labels in (test_loader):
        inputs_phobert = inputs_phobert.to(device)
        inputs_bert = inputs_bert.to(device)
        inputs_xlm = inputs_xlm.to(device)
        labels = labels.to(device)

        loss, outputs = model(  input_ids_phobert=inputs_phobert,
                                input_ids_bert=inputs_bert,
                                input_ids_xlm=inputs_xlm,
                                labels=labels)

        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs_phobert.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


def train_moe_phobert_bert_lstmcnn(dataset, num_epochs, batch_size, learning_rate, model_name='Moe_Gating'):
    date_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S ')
    if model_name == 'Moe_Gating':
        print(date_time, 'Training MOE of PhoBert, Bert and LSTMCNN Moe_Gating')
    elif model_name == 'Moe_Concate':
        print(date_time, 'Training MOE of PhoBert, Bert and LSTMCNN Moe_Concate')
    elif model_name == 'Moe_GatingGLU':
        print(date_time, 'Training MOE of PhoBert, Bert and LSTMCNN Moe_GatingGLU')

    # prepare data
    train = pd.read_csv('main/dataset/' + dataset + '/train.csv')
    
    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    tokenizer_bert = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    vocab_phobert = Dictionary()
    vocab_phobert.add_from_file('main/phobert/dict.txt')
    segmenter_phobert = VnCoreNLP('main/vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx500m')

    tokenizer_lstmcnn = load_tokenizer('main/dataset/' + dataset + '/')
    embedding_matrix_lstmcnn = load_embedding("/data/cuong/data/nlp/embedding/cc.vi.300.vec", embedding_size=300, word_index=tokenizer_lstmcnn.word_index)

    train_dataset = ReviewDataset(list(train['discriptions']), train['mapped_rating'].values,
                                    segmenter_phobert, vocab_phobert, #phobert
                                    tokenizer_bert, #bert
                                    tokenizer_lstmcnn #lstmcnn
                                    )
    valid_dataset = ReviewDataset(list(valid['discriptions']), valid['mapped_rating'].values,
                                    segmenter_phobert, vocab_phobert, #phobert
                                    tokenizer_bert, #bert
                                    tokenizer_lstmcnn #lstmcnn
                                    )

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # load model
    config_phobert = RobertaConfig.from_pretrained('main/phobert/config.json')
    config_phobert.num_labels = 2

    config_bert = BertConfig.from_pretrained('bert-base-multilingual-cased')
    config_bert.num_labels = 2

    model_lstmcnn = LSTMCNN(embedding_matrix_lstmcnn)

    if model_name == 'Moe_Gating':
        model = Moe_Gating(config_phobert, config_bert, model_lstmcnn, hidden_size=128)
    elif model_name == 'Moe_Concate':
        model = Moe_Concate(config_phobert, config_bert, model_lstmcnn, hidden_size=128)
    elif model_name == 'Moe_GatingGLU':
        model = Moe_GatingGLU(config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1)


    model.phobert.load_state_dict(torch.load('main/logs/' + dataset + '/phobert.pth'))
    model.bert.load_state_dict(torch.load('main/logs/' + dataset + '/bert.pth'))
    model.lstmcnn.load_state_dict(torch.load('main/logs/' + dataset + '/lstmcnn.pth'))

    for param in model.phobert.parameters():
        param.requires_grad = False
    for param in model.bert.parameters():
        param.requires_grad = False
    for param in model.lstmcnn.parameters():
        param.requires_grad = False

    model = model.to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('pytorch_total_params', pytorch_total_params)

    print('train dataset', train.shape)
    print("[TRAIN]", len(train_dataset))
    print("[VALID]", len(valid_dataset))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    since = time.time()
    best_acc = 0.0
    best_loss = 0.0
    best_epoch = 1
    early_stopping = 5
    cnt = 0

    # training
    print("Start training ...\n" + "==========================================================\n")
    num_epochs = num_epochs
    for epoch in range(1, num_epochs + 1):
        print('\n')

        model.train()
        running_loss = 0.0
        running_corrects = 0
        moe_alpha = np.empty((0,3), int)
        for inputs_phobert, inputs_bert, inputs_lstmcnn, labels in (train_loader):
            inputs_phobert = inputs_phobert.to(device)
            inputs_bert = inputs_bert.to(device)
            inputs_lstmcnn = inputs_lstmcnn.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            alpha, loss, outputs = model(  input_ids_phobert=inputs_phobert,
                                        input_ids_bert=inputs_bert,
                                        input_ids_lstmcnn=inputs_lstmcnn,
                                        labels=labels)
            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += error.item() * inputs_phobert.size(0)
            running_corrects += torch.sum(preds == labels.data)

            moe_alpha = np.append(moe_alpha, np.array(alpha.cpu().detach().numpy()), axis=0)
            #print('moe_alpha', moe_alpha.shape)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('[TRAIN] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        moe_alpha = np.average(moe_alpha, axis=0)
        print('epoch ', epoch, moe_alpha.shape, moe_alpha)


        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs_phobert, inputs_bert, inputs_lstmcnn, labels in (valid_loader):
            inputs_phobert = inputs_phobert.to(device)
            inputs_bert = inputs_bert.to(device)
            inputs_lstmcnn = inputs_lstmcnn.to(device)
            labels = labels.to(device)

            moe_values, loss, outputs = model(  input_ids_phobert=inputs_phobert,
                                                input_ids_bert=inputs_bert,
                                                input_ids_lstmcnn=inputs_lstmcnn,
                                                labels=labels)

            preds = torch.max(outputs, 1)[1]
            error = criterion(outputs, labels)

            running_loss += error.item() * inputs_phobert.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_dataset)
        epoch_acc = running_corrects.double() / len(valid_dataset)
        print('[VAL] epoch:{}; loss: {:.4f}; acc: {:.2f}'.format(epoch, epoch_loss, epoch_acc * 100))

        if epoch_acc > best_acc:
            cnt = 0
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_epoch = epoch

            torch.save(model.state_dict(), 'main/logs/' + dataset + '/moe.pth')


        ########################################################################
        cnt += 1
        if cnt > early_stopping:
            break
        ########################################################################


    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    print("Start testing ...\n" + "==========================================================\n")

    test = pd.read_csv('main/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDataset(list(test['discriptions']), test['mapped_rating'].values,
                                segmenter_phobert, vocab_phobert, #phobert
                                tokenizer_bert, #bert
                                tokenizer_lstmcnn, #lstmcnn
                                )
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("[TEST] ", len(test_dataset))

    model.load_state_dict(torch.load('main/logs/' + dataset + '/moe.pth'))

    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    for inputs_phobert, inputs_bert, inputs_lstmcnn, labels in (test_loader):
        inputs_phobert = inputs_phobert.to(device)
        inputs_bert = inputs_bert.to(device)
        inputs_lstmcnn = inputs_lstmcnn.to(device)
        labels = labels.to(device)

        _, loss, outputs = model(  input_ids_phobert=inputs_phobert,
                                input_ids_bert=inputs_bert,
                                input_ids_lstmcnn=inputs_lstmcnn,
                                labels=labels)

        scores = F.softmax(outputs, dim=1)
        preds = torch.max(outputs, 1)[1]
        error = criterion(outputs, labels)

        running_loss += error.item() * inputs_phobert.size(0)
        running_corrects += torch.sum(preds == labels.data)

        total_scores += list(scores.cpu().detach().numpy())
        total_preds += list(preds.cpu().numpy())
        total_labels += list(labels.data.cpu().numpy())

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))


'''
================================================================================
python -u run.py --pretrained_model="moe"       --dataset="aivivn" --batch_size=8   --model_name='Moe_Gating'
epoch  1 (3,) [0.53829947 0.52475852 0.51817843]
epoch  2 (3,) [0.55342901 0.53873552 0.51747614]
epoch  3 (3,) [0.5558085  0.54158121 0.5133908 ]
epoch  4 (3,) [0.55629187 0.545191   0.51162608]
epoch  5 (3,) [0.55569164 0.54900104 0.51250326]
epoch  6 (3,) [0.55535125 0.55144219 0.51820858]
epoch  7 (3,) [0.55606984 0.55366648 0.52770634]
epoch  8 (3,) [0.55668973 0.55431814 0.53047209]
epoch  9 (3,) [0.55790774 0.55347017 0.53258814]
epoch  10 (3,) [0.55820251 0.55392877 0.53761102]
epoch  11 (3,) [0.56113611 0.55426663 0.53899897]
epoch  12 (3,) [0.56146321 0.55354775 0.54249433]
epoch  13 (3,) [0.56391862 0.55390257 0.54285882]
epoch  14 (3,) [0.56401969 0.55307139 0.54560965]
epoch  15 (3,) [0.56432438 0.55474798 0.54712318]
epoch  16 (3,) [0.56534305 0.55265437 0.54715952]
epoch  17 (3,) [0.56506683 0.55155522 0.54683325]
epoch  18 (3,) [0.56502167 0.54971842 0.54744679]
epoch  19 (3,) [0.56899427 0.54909291 0.54835424]
epoch  20 (3,) [0.57145882 0.54898211 0.54847793]
epoch  21 (3,) [0.57185981 0.55070395 0.54973392]
epoch  22 (3,) [0.57475467 0.55070971 0.55127556]
epoch  23 (3,) [0.57755927 0.55185958 0.55367776]
epoch  24 (3,) [0.57550736 0.55039456 0.55538346]
epoch  25 (3,) [0.57734552 0.54976616 0.55410005]
epoch  26 (3,) [0.57844443 0.54859741 0.55502894]
epoch  27 (3,) [0.57846706 0.54855024 0.55561856]
epoch  28 (3,) [0.57896465 0.54727722 0.5578008 ]
epoch  29 (3,) [0.57910241 0.54798568 0.55880087]
epoch  30 (3,) [0.58073214 0.54808761 0.559048  ]


================================================================================
python -u run.py --pretrained_model="moe"      --dataset="tiki" --batch_size=32  --nrows=25000 --learning_rate=1e-4 --model_name='Moe_Gating'
epoch  1 (3,) [0.50786779 0.3961308  0.64010585]
epoch  2 (3,) [0.49400906 0.33882864 0.63698964]
epoch  3 (3,) [0.48737876 0.32346671 0.68394392]
epoch  4 (3,) [0.45065908 0.30197594 0.67774419]
epoch  5 (3,) [0.44726944 0.27200145 0.70162599]
epoch  6 (3,) [0.45523108 0.2668596  0.71278141]
epoch  7 (3,) [0.45635898 0.2451516  0.73411952]
epoch  8 (3,) [0.47014846 0.25136436 0.73808095]
epoch  9 (3,) [0.46251894 0.25315039 0.75350965]
epoch  10 (3,) [0.43655067 0.25571776 0.7691313 ]
epoch  11 (3,) [0.42485367 0.25565354 0.7630854 ]
epoch  12 (3,) [0.42862815 0.26326519 0.73977939]
epoch  13 (3,) [0.42187445 0.25549406 0.76064374]
epoch  14 (3,) [0.42622126 0.26990838 0.7704532 ]
epoch  15 (3,) [0.42247437 0.25169489 0.7366957 ]
epoch  16 (3,) [0.41783305 0.26328396 0.73466558]
epoch  17 (3,) [0.40101585 0.24335342 0.71742104]
epoch  18 (3,) [0.40279322 0.23867913 0.74744695]
epoch  19 (3,) [0.38184089 0.24779116 0.74720652]
epoch  20 (3,) [0.39026096 0.22635101 0.73316848]
epoch  21 (3,) [0.38656984 0.23515284 0.73249632]
epoch  22 (3,) [0.39112622 0.21761312 0.72909099]
epoch  23 (3,) [0.37689595 0.25964619 0.72442827]
epoch  24 (3,) [0.37727364 0.24286101 0.70259727]
epoch  25 (3,) [0.37611399 0.25685774 0.71295589]
epoch  26 (3,) [0.34829532 0.28132623 0.70260602]
epoch  27 (3,) [0.35605019 0.25911507 0.67815866]
epoch  28 (3,) [0.35537438 0.25948831 0.69632127]
epoch  29 (3,) [0.34076145 0.28867807 0.70547524]
epoch  30 (3,) [0.3450684  0.22212546 0.66940912]
epoch  31 (3,) [0.35070003 0.24591292 0.67526077]
epoch  32 (3,) [0.33635646 0.2615099  0.67049176]
epoch  33 (3,) [0.34091664 0.2557771  0.68329446]
epoch  34 (3,) [0.33066057 0.25966006 0.65718758]
epoch  35 (3,) [0.33796498 0.26030113 0.65041746]
epoch  36 (3,) [0.33248661 0.25013079 0.62665458]
epoch  37 (3,) [0.32226396 0.23582751 0.61380836]
epoch  38 (3,) [0.31162444 0.25401942 0.62437295]
epoch  39 (3,) [0.30469766 0.27316577 0.62542656]
epoch  40 (3,) [0.28960688 0.26139548 0.6373089 ]
epoch  41 (3,) [0.29605847 0.25796194 0.63541454]
epoch  42 (3,) [0.28656395 0.26418092 0.65651939]
epoch  43 (3,) [0.27824346 0.25801212 0.62794797]
epoch  44 (3,) [0.28942088 0.27161406 0.64474746]
epoch  45 (3,) [0.32180488 0.28360286 0.61353331]
epoch  46 (3,) [0.29077107 0.28817663 0.60385196]
epoch  47 (3,) [0.28039295 0.2686184  0.61044132]
epoch  48 (3,) [0.29162462 0.25844582 0.59835175]
epoch  49 (3,) [0.29392113 0.25909359 0.58594292]
epoch  50 (3,) [0.28797933 0.26271975 0.60446702]


python -u run.py --pretrained_model="moe"      --dataset="tiki" --batch_size=32  --nrows=15000 --learning_rate=1e-4 --model_name='Moe_Gating'
epoch  1 (3,) [0.47247382 0.56149355 0.53028521]
epoch  2 (3,) [0.45980858 0.54052976 0.57660647]
epoch  3 (3,) [0.47107068 0.53542993 0.56354295]
epoch  4 (3,) [0.47111728 0.50419469 0.58026441]
epoch  5 (3,) [0.47370003 0.43371318 0.58854291]
epoch  6 (3,) [0.46530287 0.42885115 0.59749908]
epoch  7 (3,) [0.45845501 0.42664759 0.5920901 ]
epoch  8 (3,) [0.47657862 0.4383622  0.59594502]
epoch  9 (3,) [0.46147703 0.42899244 0.6073714 ]
epoch  10 (3,) [0.43900298 0.39826309 0.60087179]
epoch  11 (3,) [0.43003342 0.40890571 0.63454751]
epoch  12 (3,) [0.43388839 0.39984356 0.62037999]
epoch  13 (3,) [0.42084047 0.37365582 0.61839632]
epoch  14 (3,) [0.42861564 0.38855998 0.60799833]
epoch  15 (3,) [0.39778989 0.37336076 0.63500595]
epoch  16 (3,) [0.43638578 0.36186039 0.61790096]
epoch  17 (3,) [0.42887175 0.36849439 0.57857804]
epoch  18 (3,) [0.42950537 0.36212479 0.60207694]
epoch  19 (3,) [0.42103344 0.35785442 0.60757975]
epoch  20 (3,) [0.41965253 0.33705216 0.61781161]
epoch  21 (3,) [0.39735736 0.34382999 0.60998235]
epoch  22 (3,) [0.40132771 0.3461827  0.62018811]
epoch  23 (3,) [0.40741748 0.34299042 0.58136252]
epoch  24 (3,) [0.39414638 0.359512   0.64770913]
epoch  25 (3,) [0.40106239 0.36388358 0.59565216]
epoch  26 (3,) [0.40260238 0.33149183 0.59989539]
epoch  27 (3,) [0.40592475 0.34236661 0.6078701 ]
epoch  28 (3,) [0.37967582 0.33579781 0.60503325]
epoch  29 (3,) [0.38907287 0.34603825 0.57892676]
epoch  30 (3,) [0.36892983 0.35215728 0.58795558]
epoch  31 (3,) [0.38762704 0.34896468 0.60363423]
epoch  32 (3,) [0.38076269 0.35619973 0.57839835]
epoch  33 (3,) [0.37304895 0.35803301 0.57871381]
epoch  34 (3,) [0.37342683 0.36071019 0.58441753]
epoch  35 (3,) [0.37920519 0.37252677 0.58016523]
epoch  36 (3,) [0.36614861 0.35889141 0.57905905]
epoch  37 (3,) [0.38895778 0.35264456 0.54489636]
epoch  38 (3,) [0.37581382 0.35302055 0.53419428]
epoch  39 (3,) [0.35458825 0.35902015 0.56126432]
epoch  40 (3,) [0.37921901 0.38729961 0.52910396]
epoch  41 (3,) [0.34958744 0.37720921 0.53173204]
epoch  42 (3,) [0.34924871 0.33900112 0.53718091]
epoch  43 (3,) [0.3491054  0.35833493 0.54287046]
epoch  44 (3,) [0.37104987 0.35067548 0.55690511]
epoch  45 (3,) [0.35134812 0.35418229 0.53825421]
epoch  46 (3,) [0.34845217 0.36483406 0.52191572]
epoch  47 (3,) [0.35070995 0.34269898 0.54346882]
epoch  48 (3,) [0.3584326  0.34170258 0.49844709]
epoch  49 (3,) [0.34002553 0.361689   0.54065129]
epoch  50 (3,) [0.32429345 0.35932508 0.51404151]
'''
