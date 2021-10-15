import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import XLMModel


class Xlm(nn.Module):
    def __init__(self, config_xlm, hidden_size=128, dropout_prob=0.1, maxlen=128):
        super(Xlm, self).__init__()

        self.config_xlm = config_xlm
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.num_labels = self.config_xlm.num_labels

        self.xlm = XLMModel(config_xlm)

        #self.avg_pool1d = nn.AvgPool1d(kernel_size=128, stride=1, padding=0)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(config_xlm.hidden_size, config_xlm.num_labels)

        self.xlm.init_weights()


    def forward(
        self,
        input_ids_xlm=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs_xlm = self.xlm(
            input_ids_xlm,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        ) #torch.Size([bs, seqlen, 1024])
        sequence_output = outputs_xlm[0]
        sequence_output = sequence_output.permute(0, 2, 1)
        #sequence_output = self.avg_pool1d(sequence_output)
        sequence_output = F.avg_pool1d(sequence_output, sequence_output.size(2)).squeeze(2)

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
