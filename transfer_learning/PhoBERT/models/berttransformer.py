import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, BertPreTrainedModel

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout_prob=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(1024, embedding_size)
        position = torch.arange(0, 1024, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class BertTransformer(BertPreTrainedModel):
    def __init__(self, config, num_layers=6, num_heads=8, maxlen=128, dropout_prob=0.1):
        super().__init__(config)
        self.config = config
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.maxlen = maxlen
        self.dropout_prob = dropout_prob
        self.num_labels = self.config.num_labels

        self.bert = RobertaModel(self.config)
        self.position_encoder = PositionalEncoding(self.config.hidden_size, self.dropout_prob)
        self.transformer_encoder_layers = nn.TransformerEncoderLayer(self.config.hidden_size, self.num_heads, 256, self.dropout_prob)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layers, self.num_layers)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.maxlen * self.config.hidden_size, self.num_labels)

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

        sequence_output = sequence_output * math.sqrt(self.config.hidden_size)
        position_encoded = self.position_encoder(sequence_output)
        transformer_encoded = self.transformer_encoder(position_encoded, None)
        transformer_encoded = self.dropout(transformer_encoded.view((transformer_encoded.shape[0], -1)))
        logits = self.classifier(transformer_encoded)

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