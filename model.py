import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import time
from transformers import BertPreTrainedModel, BertModel
from data_utils import get_example_rel, get_event_rel

import os
import json
import codecs


def json2dicts(jsonFile):
    data = []
    with codecs.open(jsonFile, "r", "utf-8") as f:
        for line in f:
            dic = json.loads(line)
            data.append(dic)
    return data


logger = logging.getLogger(__name__)

device = torch.device("cuda")

num = 0

def set_device(de):
    global device
    device = de


class NewModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.prototypes = nn.Embedding(config.num_labels, config.hidden_size).to(device)
        self.proto_size = config.num_labels

        self.pi = 3.14159262358979323846
        self.ratio_proto_emb = 0.4
        self.margin = 0.08
        self.r_gamma = 8
        self.wandb = None
        self.emb = 100
        self.hid = config.hidden_size
        self.re_classifier = None

        self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 3).to(device))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, example_ids=None):
        batch_size = input_ids.size(0)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = []
        for i in range(2):
            seq = outputs[2][-i]
            pooled_output += [torch.mean(seq, dim=1, keepdim=True)]
        pooled_output = torch.sum(torch.cat(pooled_output, dim=1), 1)
        pooled_output = F.relu(pooled_output)
        instance_embedding = self.dropout(pooled_output)

        torch.autograd.set_detect_anomaly(True)

        # proto_embedding = self.prototypes
        proto_embedding = self.prototypes(torch.tensor(range(0, self.proto_size)).to(device))
        instance_embedding = self.re_classifier(instance_embedding)

        logits = -self.__batch_dist__(proto_embedding, instance_embedding)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        # 极坐标
        ca_h = torch.tensor([]).to(device)
        ca_t = torch.tensor([]).to(device)
        su_h = torch.tensor([]).to(device)
        su_t = torch.tensor([]).to(device)

        rel_event_ids = get_event_rel()
        for i in labels:
            id = i.item()
            for it in rel_event_ids[id][0]:
                ca_h = torch.cat((ca_h, proto_embedding[id].unsqueeze(0)), 0)
                ca_t = torch.cat((ca_t, self.proto_emb(it[1], True, False).unsqueeze(0)), 0)
            for it in rel_event_ids[id][1]:
                ca_h = torch.cat((ca_h, self.proto_emb(it[0], True, True).unsqueeze(0)), 0)
                ca_t = torch.cat((ca_t, proto_embedding[id].unsqueeze(0)), 0)
            for it in rel_event_ids[id][2]:
                su_h = torch.cat((su_h, proto_embedding[id].unsqueeze(0)), 0)
                su_t = torch.cat((su_t, self.proto_emb(it[1], False, False).unsqueeze(0)), 0)
            for it in rel_event_ids[id][3]:
                su_h = torch.cat((su_h, proto_embedding[id].unsqueeze(0)), 0)
                su_t = torch.cat((su_t, self.proto_emb(it[0], False, True).unsqueeze(0)), 0)

        loss_p = self.func_ca(ca_h, ca_t, 0.8)
        if len(su_t) > 1:
            loss_r = self.func_su_new(su_t, torch.cat((su_t[:, 1:], su_t[:, :1]), 1), su_h)
        else:
            loss_r = self.func_su_new(su_t, su_t, su_h)

        # loss = self.major  * loss + self.p_rate * loss_p + self.r_rate * loss_r  # +ins_loss_p+ins_loss_r

        loss = loss/(2*self.loss_scale[0].exp())+self.loss_scale[0]/2
        loss += loss_p / (2 * self.loss_scale[1].exp()) + self.loss_scale[1] / 2
        loss += loss_r / (2 * self.loss_scale[2].exp()) + self.loss_scale[2] / 2


        global num
        num += 1

        outputs = (logits,) + outputs[2:]
        outputs = (loss,) + outputs

        return outputs
