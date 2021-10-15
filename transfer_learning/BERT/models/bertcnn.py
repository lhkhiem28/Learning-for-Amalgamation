import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel

class BertCNN(BertPreTrainedModel):
    def __init__(self, config, n_filters=128, kernel_sizes=[1, 3, 5], dropout_prob=0.1):
        super().__init__(config)
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_prob = dropout_prob
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.n_filters, (K, config.hidden_size)) for K in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout_prob)
        self.classifier = nn.Linear(self.n_filters * len(self.kernel_sizes), config.num_labels)

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
        sequence_output = sequence_output.unsqueeze(1)  

        convs = [conv(sequence_output) for conv in self.convs] 
        convs = [F.relu(conv).squeeze(3) for conv in convs]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  
        cat = torch.cat(pools, 1)

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