import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

class BertLSTMAttn(BertPreTrainedModel):
    def __init__(self, config, attention_type='dot', hidden_size=128, dropout_prob=0.1):
        super().__init__(config)
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.lstm = nn.LSTM(config.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(6*self.hidden_size, config.num_labels)

        self.init_weights()

    def attention(self, lstm, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(lstm, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.linear(lstm), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        
        attention = torch.bmm(lstm.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

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

        lstm, (hn, _) = self.lstm(sequence_output)
        final_hn_layer = hn.view(self.lstm.num_layers, self.lstm.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(lstm, final_hidden_state)
        avg_pool = torch.mean(lstm, 1)
        max_pool, _ = torch.max(lstm, 1)
        cat = torch.cat((avg_pool, max_pool, attention), 1)

        cat = self.dropout(cat)
        logits = self.classifier(cat)

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

class BertGRUAttn(BertPreTrainedModel):
    def __init__(self, config, attention_type='dot', hidden_size=128, dropout_prob=0.1):
        super().__init__(config)
        self.attention_type = attention_type
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.gru  = nn.GRU(config.hidden_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2*self.hidden_size, 2*self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(6*self.hidden_size, config.num_labels)

        self.init_weights()

    def attention(self, gru, final_hidden_state):
        if self.attention_type == 'dot':
            attention_weights = torch.bmm(gru, final_hidden_state.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            attention_weights = torch.bmm(self.linear(gru), final_hidden_state.unsqueeze(2)).squeeze(2)
        attention_weights = F.softmax(attention_weights, 1)
        
        attention = torch.bmm(gru.transpose(1, 2), attention_weights.unsqueeze(2)).squeeze(2)
        return attention

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

        gru, (hn) = self.gru(sequence_output)
        final_hn_layer = hn.view(self.gru.num_layers, self.gru.bidirectional+1, hn.shape[1], self.hidden_size)[-1, :, :, :]
        final_hidden_state = torch.cat([final_hn_layer[i, :, :] for i in range(final_hn_layer.shape[0])], dim=1)
        attention = self.attention(gru, final_hidden_state)
        avg_pool = torch.mean(gru, 1)
        max_pool, _ = torch.max(gru, 1)
        cat = torch.cat((avg_pool, max_pool, attention), 1)

        cat = self.dropout(cat)
        logits = self.classifier(cat)

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