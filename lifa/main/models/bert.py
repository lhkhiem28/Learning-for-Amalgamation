import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel
from transformers import RobertaModel, BertPreTrainedModel


class Bert(nn.Module):
    def __init__(self, config_bert, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Bert, self).__init__()

        self.config_bert = config_bert
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_bert.num_labels
        self.bert = BertModel(config_bert)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(config_bert.hidden_size, config_bert.num_labels)

        self.bert.init_weights()


    def forward(
        self,
        input_ids_bert=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs_bert = self.bert(
            input_ids_bert,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 768])
        sequence_output = outputs_bert[1]
        emb = sequence_output.view(sequence_output.shape[0], -1)
        outputs = (emb,)

        sequence_output = sequence_output.view(sequence_output.shape[0], -1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
