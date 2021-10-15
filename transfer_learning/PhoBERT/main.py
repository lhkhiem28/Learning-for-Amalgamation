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
from transformers import RobertaConfig
from PhoBERT.models.bert import BertFC
from PhoBERT.models.bertrnn import BertLSTM, BertGRU
from PhoBERT.models.bertcnn import BertCNN
from PhoBERT.models.bertrcnn import BertLSTMCNN, BertGRUCNN
from PhoBERT.models.bertrnnattn import BertLSTMAttn, BertGRUAttn
from PhoBERT.models.berttransformer import BertTransformer
from PhoBERT.data import ReviewDataset
from fairseq.data import Dictionary
from vncorenlp import VnCoreNLP
import sklearn

import logging
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)


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

def train(head_model, dataset, num_epochs, batch_size, learning_rate, nrows=None):
    train = pd.read_csv('PhoBERT/dataset/' + dataset + '/train.csv', nrows=nrows)
    print('train', train.shape)

    train, valid = model_selection.train_test_split(train, stratify=train['mapped_rating'], test_size=0.2, random_state=2020)

    vocab = Dictionary()
    vocab.add_from_file('PhoBERT/phobert/dict.txt')
    segmenter = VnCoreNLP('PhoBERT/vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx500m')

    train_dataset = ReviewDataset(list(train['discriptions']), train['mapped_rating'].values, segmenter, vocab)
    valid_dataset = ReviewDataset(list(valid['discriptions']), valid['mapped_rating'].values, segmenter, vocab)

    train_loader = utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    config = RobertaConfig.from_pretrained('PhoBERT/phobert/config.json')
    config.num_labels = 2

    if head_model == 'fc':
        model = BertFC(config=config, dropout_prob=0.1)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)

    elif head_model == 'lstm':
        model = BertLSTM(config=config, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'gru':
        model = BertGRU(config=config, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'lstm-attn':
        model = BertLSTMAttn(config=config, hidden_size=128, dropout_prob=0.1, attention_type='general')

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)

    elif head_model == 'gru-attn':
        model = BertGRUAttn(config=config, hidden_size=128, dropout_prob=0.1, attention_type='general')
    elif head_model == 'lstm-cnn':
        model = BertLSTMCNN(config=config, hidden_size=128, dropout_prob=0.1)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        
    elif head_model == 'gru-cnn':
        model = BertGRUCNN(config=config, hidden_size=128, dropout_prob=0.1)
    elif head_model == 'cnn':
        model = BertCNN(config=config, n_filters=128, kernel_sizes=[1, 3, 5], dropout_prob=0.1)
    elif head_model == 'transformer':
        model = BertTransformer(config=config, num_layers=2, num_heads=8, maxlen=128, dropout_prob=0.1)

    model = model.from_pretrained('PhoBERT/phobert/model.bin', config=config)
    model = model.to(device)
    #print(model)

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
            torch.save(model.state_dict(), 'PhoBERT/logs/' + dataset + '/bert_{}.pth'.format(head_model))

        cnt += 1
        if cnt > early_stopping:
            break


    time_elapsed = time.time() - since
    print('\nTraining time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('[VALID] epoch:{:2} - loss:{:.4f} - acc:{:.2f}'
        .format(best_epoch, best_loss, best_acc * 100))


    # testing
    test = pd.read_csv('PhoBERT/dataset/' + dataset + '/test.csv')
    test_dataset = ReviewDataset(list(test['discriptions']), test['mapped_rating'].values, segmenter, vocab)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load('PhoBERT/logs/' + dataset + '/bert_{}.pth'.format(head_model)))
    model = model.to(device)
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total_scores = []
    total_preds = []
    total_labels = []
    #for inputs, labels in tqdm.tqdm(test_loader):
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

    total_loss = running_loss / len(test_dataset)
    total_acc = running_corrects.double() / len(test_dataset)
    precision = sklearn.metrics.precision_score(total_labels, total_preds)
    recall = sklearn.metrics.recall_score(total_labels, total_preds)
    f1 = sklearn.metrics.f1_score(total_labels, total_preds)
    auc = sklearn.metrics.roc_auc_score(onehot_labels(total_labels), total_scores)
    print('[TEST]  loss:{:.4f} - acc:{:.2f} - precision:{:.4f} - recall:{:.4f} - f1:{:.4f} - auc:{:.4f}'
          .format(total_loss, total_acc * 100, precision, recall, f1, auc))

    #evaluate(model, criterion=nn.CrossEntropyLoss(), dataset=test_dataset, batch_size=batch_size)
