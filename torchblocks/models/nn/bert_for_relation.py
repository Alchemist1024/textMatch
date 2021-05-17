import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class REBERT(BertPreTrainedModel):
    def __init__(self, config):
        super(REBERT, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, 0.1)
        self.label_classifier = FCLayer(config.hidden_size * 3, config.num_labels, 0.1, use_activation=False)

    def entity_average(self, hidden_output, e_mask):
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b,1,j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # (batch_size,1)
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(
            1)  # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)
        # Dropout -> tanh -> fc_layer
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)
        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)
