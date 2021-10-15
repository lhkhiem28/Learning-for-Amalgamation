import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

class BertLSTMCNN(BertPreTrainedModel):
    def __init__(self, config, hidden_size=128, dropout_prob=0.1):
        super().__init__(config)
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.lstm = nn.LSTM(config.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(config.hidden_size + 2*self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        lstm, _ = self.lstm(sequence_output)
        cat = torch.cat((lstm[:, :, :self.hidden_size], sequence_output, lstm[:, :, self.hidden_size:]), 2)
        linear = F.tanh(self.fc(cat)).permute(0, 2, 1)
        pool = F.max_pool1d(linear, linear.shape[2]).squeeze(2)
        
        pool = self.dropout(pool)
        logits = self.classifier(pool)

        outputs = (logits,) + outputs[2:] 

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

class BertGRUCNN(BertPreTrainedModel):
    def __init__(self, config, hidden_size=128, dropout_prob=0.1):
        super().__init__(config)
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.gru  = nn.GRU(config.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(config.hidden_size + 2*self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        gru, _ = self.gru(sequence_output)
        cat = torch.cat((gru[:, :, :self.hidden_size], sequence_output, gru[:, :, self.hidden_size:]), 2)
        linear = F.tanh(self.fc(cat)).permute(0, 2, 1)
        pool = F.max_pool1d(linear, linear.shape[2]).squeeze(2)
        
        pool = self.dropout(pool)
        logits = self.classifier(pool)

        outputs = (logits,) + outputs[2:] 

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs