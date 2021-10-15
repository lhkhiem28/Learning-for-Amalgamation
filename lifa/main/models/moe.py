import torch
import torch.nn as nn
import torch.nn.functional as F
from main.models.bert import *
from main.models.phobert import *
from main.models.xlm import *
from main.models import acts



class Moe_Gating(nn.Module):
    def __init__(self, config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Moe_Gating, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert

        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels

        self.experts = 3
        self.phobert = PhoBert(self.config_phobert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.bert = Bert(self.config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.lstmcnn = model_lstmcnn

        #self.emb_size = 256
        self.emb_size = 512
        #self.emb_size = 1024
        self.conv_phobert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_bert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_lstmcnn = nn.Conv1d(256, self.emb_size, kernel_size=1, bias=False)

        self.gate = nn.Linear(self.emb_size*self.experts, self.experts)
        #self.gate_bn = nn.BatchNorm1d(self.experts) #XXX

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.emb_size, config_bert.num_labels)


    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        input_ids_lstmcnn=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get embedding features
        _, _, emb_phobert = self.phobert(input_ids_phobert, labels=labels) #torch.Size([bs, 768])
        _, _, emb_bert = self.bert(input_ids_bert, labels=labels) #torch.Size([bs, 768])
        _, emb_lstmcnn, _ = self.lstmcnn(input_ids_lstmcnn, labels=labels) #torch.Size([bs, 1024])

        #print('emb_phobert', emb_phobert.shape)
        #print('emb_bert', emb_bert.shape)
        #print('emb_lstmcnn', emb_lstmcnn.shape)

        # conv to get same feature dimensionality
        emb_phobert = torch.unsqueeze(emb_phobert, 2) #(bs,768,1)
        emb_bert = torch.unsqueeze(emb_bert, 2) #(bs,768,1)
        emb_lstmcnn = torch.unsqueeze(emb_lstmcnn, 2) #(bs,1024,1)

        emb_phobert = self.conv_phobert(emb_phobert) #(bs,emb_size,1)
        emb_bert = self.conv_bert(emb_bert) #(bs,emb_size,1)
        emb_lstmcnn = self.conv_lstmcnn(emb_lstmcnn) #(bs,emb_size,1)

        emb_phobert = emb_phobert.view(emb_phobert.shape[0], -1) #(bs,emb_size)
        emb_bert = emb_bert.view(emb_bert.shape[0], -1) #(bs,emb_size)
        emb_lstmcnn = emb_lstmcnn.view(emb_lstmcnn.shape[0], -1) #(bs,emb_size)

        # moe with sigmoid/softmax
        inputs_gate = torch.cat([emb_phobert, emb_bert, emb_lstmcnn], dim=1) #torch.Size([bs, emb_size*2]

        inputs_gate = inputs_gate.detach()
        outputs_gate = self.gate(inputs_gate.float())


        #outputs_gate_softmax = F.sigmoid(outputs_gate) #torch.Size([8, 2])
        #outputs_gate_softmax = F.gumbel_softmax(outputs_gate, tau=0.01, hard=False)
        #outputs_gate_softmax = F.gumbel_softmax(outputs_gate, tau=0.1, hard=False)
        #outputs_gate_softmax = F.gumbel_softmax(outputs_gate, tau=10, hard=False)
        outputs_gate_softmax = F.gumbel_softmax(outputs_gate, tau=100, hard=False)

        #print('outputs_gate_softmax after softmax', outputs_gate_softmax)
        #outputs_gate_softmax = torch.clamp(outputs_gate_softmax, min=0.1, max=2.0)
        #print('outputs_gate_softmax after clamp with 0.1', outputs_gate_softmax)
        #outputs_gate_softmax = F.normalize(outputs_gate_softmax, p=2, dim=1, eps=1e-12, out=None)
        #print('outputs_gate_softmax after normalizing', outputs_gate_softmax)


        sequence_output = torch.stack([emb_phobert, emb_bert, emb_lstmcnn], dim=-2) # bs x emb_size x #experts
        sequence_output = torch.sum(outputs_gate_softmax.unsqueeze(-1) * sequence_output, dim=-2) # bs x emb_size #torch.Size([8, 512])

        # classifier
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs


        outputs = (outputs_gate_softmax,) + outputs ########


        return outputs


class Moe_GatingGLU(nn.Module):
    def __init__(self, config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Moe_GatingGLU, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert

        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels

        self.experts = 3
        self.phobert = PhoBert(self.config_phobert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.bert = Bert(self.config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.lstmcnn = model_lstmcnn

        self.gating_module = nn.Sequential(
            nn.Linear(1792, 1792 // 8, bias=False),
            acts.Swish(),
            nn.Linear(1792 // 8, 1792, bias=False),
            acts.Sigmoid(),
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(1792, config_bert.num_labels)


    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        input_ids_lstmcnn=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get embedding features
        _, _, emb_phobert = self.phobert(input_ids_phobert, labels=labels) #torch.Size([bs, 768])
        _, _, emb_bert = self.bert(input_ids_bert, labels=labels) #torch.Size([bs, 768])
        _, emb_lstmcnn, _ = self.lstmcnn(input_ids_lstmcnn, labels=labels) #torch.Size([bs, 1024])

        #print('emb_phobert', emb_phobert.shape)
        #print('emb_bert', emb_bert.shape)
        #print('emb_lstmcnn', emb_lstmcnn.shape)

        # gating
        cat = torch.cat(
            (
                emb_phobert,
                emb_bert,
                emb_lstmcnn,
            ), 1
        )

        sequence_output = self.gating_module(cat) * cat

        # classifier
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class Moe_MGCNN(nn.Module):
    def __init__(self, config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Moe_MGCNN, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert

        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels

        self.experts = 3
        self.phobert = PhoBert(self.config_phobert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.bert = Bert(self.config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.lstmcnn = model_lstmcnn

        #self.emb_size = 512
        self.emb_size = 256
        #self.emb_size = 768
        self.conv_phobert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_bert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_lstmcnn = nn.Conv1d(256, self.emb_size, kernel_size=1, bias=False)

        self.gate = nn.Linear(self.emb_size*self.experts, self.experts)
        #self.gate_bn = nn.BatchNorm1d(self.experts) #XXX

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.emb_size*self.experts, config_bert.num_labels)


    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        input_ids_lstmcnn=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get embedding features
        _, _, emb_phobert = self.phobert(input_ids_phobert, labels=labels) #torch.Size([bs, 768])
        _, _, emb_bert = self.bert(input_ids_bert, labels=labels) #torch.Size([bs, 768])
        _, emb_lstmcnn, _ = self.lstmcnn(input_ids_lstmcnn, labels=labels) #torch.Size([bs, 1024])

        #print('emb_phobert', emb_phobert.shape)
        #print('emb_bert', emb_bert.shape)
        #print('emb_lstmcnn', emb_lstmcnn.shape)

        # conv to get same feature dimensionality
        emb_phobert = torch.unsqueeze(emb_phobert, 2) #(bs,768,1)
        emb_bert = torch.unsqueeze(emb_bert, 2) #(bs,768,1)
        emb_lstmcnn = torch.unsqueeze(emb_lstmcnn, 2) #(bs,1024,1)

        emb_phobert = self.conv_phobert(emb_phobert) #(bs,emb_size,1)
        emb_bert = self.conv_bert(emb_bert) #(bs,emb_size,1)
        emb_lstmcnn = self.conv_lstmcnn(emb_lstmcnn) #(bs,emb_size,1)

        emb_phobert = emb_phobert.view(emb_phobert.shape[0], -1) #(bs,emb_size)
        emb_bert = emb_bert.view(emb_bert.shape[0], -1) #(bs,emb_size)
        emb_lstmcnn = emb_lstmcnn.view(emb_lstmcnn.shape[0], -1) #(bs,emb_size)

        # moe with concatenation
        sequence_output = torch.cat(
            (
                emb_phobert,
                emb_bert,
                emb_lstmcnn,
            ), 1
        )

        # classifier
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class Moe_Average(nn.Module):
    def __init__(self, config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Moe_Average, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert

        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels

        self.experts = 3
        self.phobert = PhoBert(self.config_phobert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.bert = Bert(self.config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.lstmcnn = model_lstmcnn

        self.emb_size = 512
        self.conv_phobert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_bert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_lstmcnn = nn.Conv1d(256, self.emb_size, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.emb_size, config_bert.num_labels)


    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        input_ids_lstmcnn=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get embedding features
        _, _, emb_phobert = self.phobert(input_ids_phobert, labels=labels) #torch.Size([bs, 768])
        _, _, emb_bert = self.bert(input_ids_bert, labels=labels) #torch.Size([bs, 768])
        _, emb_lstmcnn, _ = self.lstmcnn(input_ids_lstmcnn, labels=labels) #torch.Size([bs, 1024])

        #print('emb_phobert', emb_phobert.shape)
        #print('emb_bert', emb_bert.shape)
        #print('emb_lstmcnn', emb_lstmcnn.shape)

        # conv to get same feature dimensionality
        emb_phobert = torch.unsqueeze(emb_phobert, 2) #(bs,768,1)
        emb_bert = torch.unsqueeze(emb_bert, 2) #(bs,768,1)
        emb_lstmcnn = torch.unsqueeze(emb_lstmcnn, 2) #(bs,1024,1)

        emb_phobert = self.conv_phobert(emb_phobert) #(bs,emb_size,1)
        emb_bert = self.conv_bert(emb_bert) #(bs,emb_size,1)
        emb_lstmcnn = self.conv_lstmcnn(emb_lstmcnn) #(bs,emb_size,1)

        emb_phobert = emb_phobert.view(emb_phobert.shape[0], -1) #(bs,emb_size)
        emb_bert = emb_bert.view(emb_bert.shape[0], -1) #(bs,emb_size)
        emb_lstmcnn = emb_lstmcnn.view(emb_lstmcnn.shape[0], -1) #(bs,emb_size)

        # moe with concatenation
        sequence_output = emb_phobert/3 + emb_bert/3 + emb_lstmcnn/3

        # classifier
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class Moe_Concate(nn.Module):
    def __init__(self, config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Moe_Concate, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert

        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels

        self.experts = 3
        self.phobert = PhoBert(self.config_phobert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.bert = Bert(self.config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.lstmcnn = model_lstmcnn

        self.emb_size = 512
        self.conv_phobert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_bert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_lstmcnn = nn.Conv1d(256, self.emb_size, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(768*2 + 256, config_bert.num_labels)


    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        input_ids_lstmcnn=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get embedding features
        _, _, emb_phobert = self.phobert(input_ids_phobert, labels=labels) #torch.Size([bs, 768])
        _, _, emb_bert = self.bert(input_ids_bert, labels=labels) #torch.Size([bs, 768])
        _, emb_lstmcnn, _ = self.lstmcnn(input_ids_lstmcnn, labels=labels) #torch.Size([bs, 1024])

        #print('emb_phobert', emb_phobert.shape)
        #print('emb_bert', emb_bert.shape)
        #print('emb_lstmcnn', emb_lstmcnn.shape)

        '''
        # conv to get same feature dimensionality
        emb_phobert = torch.unsqueeze(emb_phobert, 2) #(bs,768,1)
        emb_bert = torch.unsqueeze(emb_bert, 2) #(bs,768,1)
        emb_lstmcnn = torch.unsqueeze(emb_lstmcnn, 2) #(bs,1024,1)

        emb_phobert = self.conv_phobert(emb_phobert) #(bs,emb_size,1)
        emb_bert = self.conv_bert(emb_bert) #(bs,emb_size,1)
        emb_lstmcnn = self.conv_lstmcnn(emb_lstmcnn) #(bs,emb_size,1)

        emb_phobert = emb_phobert.view(emb_phobert.shape[0], -1) #(bs,emb_size)
        emb_bert = emb_bert.view(emb_bert.shape[0], -1) #(bs,emb_size)
        emb_lstmcnn = emb_lstmcnn.view(emb_lstmcnn.shape[0], -1) #(bs,emb_size)
        '''

        # moe with concatenation
        sequence_output = torch.cat(
            (
                emb_phobert,
                emb_bert,
                emb_lstmcnn,
            ), 1
        )

        # classifier
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs



class Moe_SE(nn.Module):
    def __init__(self, config_phobert, config_bert, model_lstmcnn, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Moe_SE, self).__init__()

        self.config_phobert = config_phobert
        self.config_bert = config_bert

        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels

        self.experts = 3
        self.phobert = PhoBert(self.config_phobert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.bert = Bert(self.config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128)
        self.lstmcnn = model_lstmcnn

        self.emb_size = 512
        self.conv_phobert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_bert = nn.Conv1d(768, self.emb_size, kernel_size=1, bias=False)
        self.conv_lstmcnn = nn.Conv1d(256, self.emb_size, kernel_size=1, bias=False)

        self.se_module = nn.Sequential(
            nn.Linear(self.experts, 3, bias=False),
            acts.Swish(),
            nn.Linear(3, self.experts, bias=False),
            acts.Sigmoid(),
        )

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.emb_size*self.experts, config_bert.num_labels)


    def forward(
        self,
        input_ids_phobert=None,
        input_ids_bert=None,
        input_ids_lstmcnn=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # get embedding features
        _, _, emb_phobert = self.phobert(input_ids_phobert, labels=labels) #torch.Size([bs, 768])
        _, _, emb_bert = self.bert(input_ids_bert, labels=labels) #torch.Size([bs, 768])
        _, emb_lstmcnn, _ = self.lstmcnn(input_ids_lstmcnn, labels=labels) #torch.Size([bs, 1024])

        #print('emb_phobert', emb_phobert.shape)
        #print('emb_bert', emb_bert.shape)
        #print('emb_lstmcnn', emb_lstmcnn.shape)

        # conv to get same feature dimensionality
        emb_phobert = torch.unsqueeze(emb_phobert, 2) #(bs,768,1)
        emb_bert = torch.unsqueeze(emb_bert, 2) #(bs,768,1)
        emb_lstmcnn = torch.unsqueeze(emb_lstmcnn, 2) #(bs,1024,1)

        emb_phobert = self.conv_phobert(emb_phobert) #(bs,emb_size,1)
        emb_bert = self.conv_bert(emb_bert) #(bs,emb_size,1)
        emb_lstmcnn = self.conv_lstmcnn(emb_lstmcnn) #(bs,emb_size,1)

        emb_phobert = emb_phobert.view(emb_phobert.shape[0], -1) #(bs,emb_size)
        emb_bert = emb_bert.view(emb_bert.shape[0], -1) #(bs,emb_size)
        emb_lstmcnn = emb_lstmcnn.view(emb_lstmcnn.shape[0], -1) #(bs,emb_size)

        # moe with SE
        stack = torch.stack(
            [
                emb_phobert,
                emb_bert,
                emb_lstmcnn,
            ], -2
        )
        pool = F.adaptive_avg_pool1d(stack, 1)
        pool = pool.view(pool.size(0), -1)

        se = self.se_module(pool)
        sequence_output = se.unsqueeze(-1) * stack

        # classifier
        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        #outputs = (logits,) + outputs[2:]
        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
