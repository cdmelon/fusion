# paie model
import torch
import torch.nn as nn
import numpy as np
from transformers.models.bart.modeling_bart import BartModel, BartPretrainedModel, BartForConditionalGeneration
from utils import hungarian_matcher, get_best_span, get_best_span_simple
import torch.nn.functional as F
import copy
import re
import random
from transformers import BartTokenizerFast
from torchmetrics import HingeLoss


class PAIE(BartPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # self.model = BartForConditionalGeneration(config)
        self.model = BartModel(config)

        self.w_context_start = nn.Parameter(torch.rand(config.d_model, ))
        self.w_context_end = nn.Parameter(torch.rand(config.d_model, ))

        self.w_prompt_start = nn.Parameter(torch.rand(500, ))
        self.w_prompt_end = nn.Parameter(torch.rand(500, ))

        self.w_context_start_t = nn.Linear(config.d_model * 4, config.d_model, bias=False)
        self.w_context_end_t = nn.Linear(config.d_model * 4, config.d_model, bias=False)

        # self.w_p_start = nn.Embedding(config.d_model*4,config.d_model).to(self.config.device)
        # self.w_p_end = nn.Embedding(config.d_model * 4, config.d_model).to(self.config.device)
        # self.w_p_start = nn.Linear(config.d_model * 4, config.d_model,  bias=False)

        # self.final_logits_bias = torch.zeros((1, self.model.shared.num_embeddings+2)).to(self.config.device)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings + 3)))
        # model_temp = BartForConditionalGeneration.from_pretrained("./bart", from_tf=bool('.ckpt' in "./bart"),
        #                                                      config="./bart")

        # self.lm_head = copy.deepcopy(model_temp.get_output_embeddings())
        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # self.model._init_weights(self.w_prompt_start)
        # self.model._init_weights(self.w_prompt_end)
        self.loss_fct = nn.CrossEntropyLoss(reduction='sum')

        # self.loss_ratio = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.loss_ratio = 0.5
        self.random_ratio = 0
        self.num_mask = 1
        self.constrained_gen = True
        self.cons_loss = True

        self.dropout = nn.Dropout(0.8)

        self.mat_label4token = nn.Embedding(500, 500).to(self.config.device)

    def get_loss(self):
        return self.loss_ratio

    def set_tokenizer(self, tok):
        self.tokenizer = tok

    def set_step(self, curr, max):
        self.curr_step = curr
        self.max_step = max

    def set_loss_rate(self, rate):
        if rate == 0 or rate == 1:
            self.loss_ratio = nn.Parameter(torch.tensor(rate), requires_grad=False)
        else:
            self.loss_ratio = nn.Parameter(torch.tensor(rate), requires_grad=False)

    def label2vec(self, label, label_size):
        label_vec = torch.zeros([label_size]).to(self.config.device)
        label_vec[label] = 1
        return label_vec

    def token_energy_function(self, token_embed, token_y):
        temp = self.mat_label4token(torch.tensor(range(0, 500)).to(self.config.device))
        token_local_energy_temp = torch.matmul(temp, token_embed.float())
        token_local_energy = torch.sum(torch.mul(token_y, token_local_energy_temp.transpose(0, 1)))
        seq_length = token_y.size(0)
        token_label_energy = torch.sum(torch.matmul(token_y, temp))
        token_energy = token_local_energy + token_label_energy
        return token_energy

    def __dist__(self, x, y, dim):
        dist = torch.pow(x - y, 2).sum(dim)
        # dist = torch.where(torch.isnan(dist), torch.full_like(dist, 1e-8), dist)
        return dist

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def convert_logit(self, gen_logit, enc_input_id):
        lm_logits = torch.full((gen_logit.size(0), self.vocab_size), fill_value=-1000,
                               dtype=gen_logit.dtype).to(gen_logit.device)
        index = enc_input_id.unsqueeze(dim=0).expand_as(gen_logit)
        lm_logits.scatter_(dim=1, index=index, src=gen_logit)
        return lm_logits

    def forward(
            self,
            enc_input_ids=None,
            enc_mask_ids=None,
            dec_prompt_ids=None,
            dec_prompt_mask_ids=None,
            arg_joint_prompts=None,
            target_info=None,
            old_tok_to_new_tok_indexs=None,
            arg_list=None,
            event_trigger=None,
            dec_prompt_text=None
    ):

        if not self.constrained_gen:
            prompt_split=list()
            for i in range(len(target_info)):
                prompt_slots = dict()
                for arg_temp in target_info[i].keys():
                    if self.training:
                        if len(target_info[i][arg_temp]["text"]) != 0:
                            text = " and ".join(target_info[i][arg_temp]["text"])
                            dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " " + text)
                        else:
                            dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " <s>")
                    else:
                        prompt_slots[arg_temp]=dict()
                        temp_sent = dec_prompt_text[i].split(arg_temp + " <mask>")
                        prompt_slots[arg_temp]["sent"]=temp_sent[0] + arg_temp + " <mask>"
                        # dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp)

                        dec_prompt = self.tokenizer(prompt_slots[arg_temp]["sent"])
                        dec_prompt_id, dec_prompt_mask_id = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                        while len(dec_prompt_id) < 100:
                            dec_prompt_id.append(self.tokenizer.pad_token_id)
                            dec_prompt_mask_id.append(0)
                        prompt_slots[arg_temp]["dec_id"] = dec_prompt_id
                        prompt_slots[arg_temp]["dec_mask_id"] = dec_prompt_mask_id

                        loc = prompt_slots[arg_temp]["sent"].find(arg_temp)
                        prompt_slots[arg_temp]["p_e"]=dec_prompt.char_to_token(loc + len(arg_temp) - 1) + 1
                        prompt_slots[arg_temp]["p_s"]=dec_prompt.char_to_token(loc)

                        loc = prompt_slots[arg_temp]["sent"].find("<mask>")
                        prompt_slots[arg_temp]["m_e"] = dec_prompt.char_to_token(loc) + 1
                        prompt_slots[arg_temp]["m_s"] = dec_prompt.char_to_token(loc)

                prompt_split.append(prompt_slots)

            if self.training:
                dec_prompt_ids = []
                dec_prompt_mask_ids = []
                dec_role = []
                for i in range(len(dec_prompt_text)):
                    dec_prompt = self.tokenizer(dec_prompt_text[i])
                    dec_prompt_id, dec_prompt_mask_id = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                    while len(dec_prompt_id) < 100:
                        dec_prompt_id.append(self.tokenizer.pad_token_id)
                        dec_prompt_mask_id.append(0)
                    dec_prompt_ids.append(dec_prompt_id)
                    dec_prompt_mask_ids.append(dec_prompt_mask_id)
                    dec_temp = {}
                    for arg_temp in arg_list[i]:
                        prompt_slots = {
                            "tok_s": list(), "tok_e": list(),
                        }
                        for matching_result in re.finditer(r'\b' + re.escape(arg_temp) + r'\b', dec_prompt_text[i]):
                            char_idx_s, char_idx_e = matching_result.span()
                            char_idx_e -= 1
                            tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                            tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                            prompt_slots["tok_s"].append(tok_prompt_s)
                            prompt_slots["tok_e"].append(tok_prompt_e)
                        dec_temp[arg_temp] = prompt_slots
                    dec_role.append(dec_temp)
                dec_prompt_ids = torch.tensor(dec_prompt_ids).to(self.config.device)
                dec_prompt_mask_ids = torch.tensor(dec_prompt_mask_ids).to(self.config.device)
                arg_joint_prompts = dec_role
                arg_joint_mask = dec_role



        else:
            for i in range(len(target_info)):
                for arg_temp in target_info[i].keys():
                    if self.training:
                        if random.random() > ((self.curr_step / self.max_step)*(1-self.random_ratio) + self.random_ratio):

                            for j in range(len(target_info[i][arg_temp]["text"])):
                                dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " " +
                                                                                target_info[i][arg_temp]["text"][j], 1)
                            # if len(target_info[i][arg_temp]["text"]) == 1:
                            #     dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " " +
                            #                                                     target_info[i][arg_temp]["text"][0], 1)
                            # elif len(target_info[i][arg_temp]["text"]) > 1:
                            #     dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " " +
                            #                                                     target_info[i][arg_temp]["text"][0], 1)
                            #     dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " " +
                            #                                                     target_info[i][arg_temp]["text"][1], 1)
                            dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>",
                                                                            arg_temp + "<s>")
                            #
                            #
                            # if len(target_info[i][arg_temp]["text"]) != 0:
                            #     # # if random.random() > (1-self.curr_step/self.max_step):
                            #     # #     dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " <gen>")
                            #     # # else:
                            #     # if "<none>" not in dec_prompt_text[i]:
                            #     #     text = " and ".join(target_info[i][arg_temp]["text"])
                            #     #     dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>",
                            #     #                                                     arg_temp + " " + text)
                            #     #     continue
                            #
                            #     if len(target_info[i][arg_temp]["text"]) == 1:
                            #         dec_prompt_text[i]=dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " " + target_info[i][arg_temp]["text"][0])
                            #         dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <none>",
                            #                                                         arg_temp + " <s>")
                            #     else:
                            #         dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>",
                            #                                                         arg_temp + " " +
                            #                                                         target_info[i][arg_temp]["text"][0])
                            #         dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <none>",
                            #                                                         arg_temp + " " +
                            #                                                         target_info[i][arg_temp]["text"][1])
                            # else:
                            #     # dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + " <none>")
                            #     dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>",
                            #                                                     arg_temp + " <s>")
                #         else:
                #             # text = ""
                #             # for _ in range(self.num_mask):
                #             #     text += "<mask>"
                #             dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + "<mask>")
                #             # dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <none>", arg_temp + "  <mask>")
                #     else:
                #         # text = ""
                #         # for _ in range(self.num_mask):
                #         #     text += "<mask>"
                #         dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <mask>", arg_temp + "<mask>")
                #         # dec_prompt_text[i] = dec_prompt_text[i].replace(arg_temp + " <none>", arg_temp + "  <mask>")
                #
                # assert dec_prompt_text[i].find(arg_temp + " <mask>") == -1
                # assert dec_prompt_text[i].find(arg_temp + " <none>") == -1

            dec_prompt_ids = []
            dec_prompt_mask_ids = []
            dec_role = []
            dec_mask = []
            for i in range(len(dec_prompt_text)):
                dec_prompt = self.tokenizer(dec_prompt_text[i])
                dec_prompt_id, dec_prompt_mask_id = dec_prompt["input_ids"], dec_prompt["attention_mask"]
                while len(dec_prompt_id) < 100:
                    dec_prompt_id.append(self.tokenizer.pad_token_id)
                    dec_prompt_mask_id.append(0)
                dec_prompt_ids.append(dec_prompt_id)
                dec_prompt_mask_ids.append(dec_prompt_mask_id)
                dec_temp={}
                dec_mask_temp={}
                for arg_temp in arg_list[i]:
                    prompt_slots = {
                        "tok_s": list(), "tok_e": list(),
                    }
                    for matching_result in re.finditer(r'\b' + re.escape(arg_temp) + r'\b', dec_prompt_text[i]):
                        char_idx_s, char_idx_e = matching_result.span()
                        char_idx_e -= 1
                        tok_prompt_s = dec_prompt.char_to_token(char_idx_s)
                        tok_prompt_e = dec_prompt.char_to_token(char_idx_e) + 1
                        prompt_slots["tok_s"].append(tok_prompt_s)
                        prompt_slots["tok_e"].append(tok_prompt_e)
                    dec_temp[arg_temp]=prompt_slots

                    prompt_slots = {
                        "tok_s": list(), "tok_e": list(), "has_ans": True
                    }

                    for j in range(len(target_info[i][arg_temp]["text"])):
                        # assert self.training is True
                        text = target_info[i][arg_temp]["text"][j]
                        loc = dec_prompt_text[i].find(arg_temp + " " + text)
                        if loc == -1:
                            break
                        prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp)-1)+1)
                        prompt_slots["tok_e"].append(dec_prompt.char_to_token(loc + len(arg_temp + " " + text) - 1) + 1)

                    # if len(target_info[i][arg_temp]["text"]) !=0:
                    #
                    #     text = target_info[i][arg_temp]["text"][0]
                    #     # loc = dec_prompt_text[i].find(" and ".join(target_info[i][arg_temp]["text"]))
                    #     loc = dec_prompt_text[i].find(arg_temp + " " + text)
                    #     if loc != -1:
                    #         # prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc))
                    #         # prompt_slots["tok_e"].append(dec_prompt.char_to_token(loc+len(text)-1)+1)
                    #         prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp)-1)+1)
                    #         prompt_slots["tok_e"].append(dec_prompt.char_to_token(loc + len(arg_temp + " " + text) - 1) + 1)
                    #     if len(target_info[i][arg_temp]["text"])>1:
                    #         text = target_info[i][arg_temp]["text"][1]
                    #         # loc = dec_prompt_text[i].find(" and ".join(target_info[i][arg_temp]["text"]))
                    #         loc = dec_prompt_text[i].find(arg_temp + " " + text)
                    #         if loc != -1:
                    #             prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp) - 1) + 1)
                    #             prompt_slots["tok_e"].append(
                    #                 dec_prompt.char_to_token(loc + len(arg_temp + " " + text) - 1) + 1)

                    if dec_prompt_text[i].find(arg_temp + "<s>") != -1:
                        assert self.training is True
                        for j in re.finditer(arg_temp + "<s>",dec_prompt_text[i]):
                            prompt_slots["tok_e"].append(dec_prompt.char_to_token(j.regs[0][1]-1)+1)
                            prompt_slots["tok_s"].append(dec_prompt.char_to_token(j.regs[0][0]+len(arg_temp)-1)+1)
                    if dec_prompt_text[i].find(arg_temp + " <mask>") != -1:
                        for j in re.finditer(arg_temp + " <mask>",dec_prompt_text[i]):
                            prompt_slots["tok_e"].append(dec_prompt.char_to_token(j.regs[0][1]-1)+1)
                            prompt_slots["tok_s"].append(dec_prompt.char_to_token(j.regs[0][0]+len(arg_temp)-1)+1)
                        prompt_slots["has_ans"] = False
                        # loc = dec_prompt_text[i].find(arg_temp + " <s>")
                        # prompt_slots["tok_e"].append(dec_prompt.char_to_token(loc + len(arg_temp + " <s>") - 1) + 1)
                        # prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp + " <s>") - 1))
                    # if self.num_mask == 0:
                    #     prompt_slots["tok_s"].append(dec_temp[arg_temp]["tok_e"][0])
                    #     prompt_slots["tok_e"].append(dec_temp[arg_temp]["tok_e"][0]+1)
                    #     prompt_slots["has_ans"] = False
                    # if dec_prompt_text[i].find(arg_temp + "<mask>") != -1:
                    #     loc = dec_prompt_text[i].find(arg_temp + "<mask>")
                    #     assert loc != -1
                    #     end = dec_prompt.char_to_token(loc + len(arg_temp + "<mask>") - 1) + 1
                    #     end += self.num_mask-1
                    #     prompt_slots["tok_e"].append(end)
                    #     prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp+"<mask>") - 1))
                    #
                    #     loc = dec_prompt_text[i].find(arg_temp + "  <mask>")
                    #     if loc != -1:
                    #         end = dec_prompt.char_to_token(loc + len(arg_temp + "  <mask>") - 1) + 1
                    #         end += self.num_mask - 1
                    #         prompt_slots["tok_e"].append(end)
                    #         prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp + "  <mask>") - 1))
                    #
                    #     prompt_slots["has_ans"] = False
                    #
                    # if loc != -1:
                    #     end = dec_prompt.char_to_token(loc + len(arg_temp+" <mask>") - 1) + 1
                    #     end += self.num_mask-1
                    #     prompt_slots["tok_e"].append(end)
                    #     prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp+" <mask>") - 1))
                    # loc = dec_prompt_text[i].find(arg_temp + " <none>")
                    # if loc != -1:
                    #     loc = dec_prompt_text[i].find(arg_temp + " <none>")
                    #     prompt_slots["tok_e"].append(dec_prompt.char_to_token(loc + len(arg_temp + " <none>") - 1) + 1)
                    #     prompt_slots["tok_s"].append(dec_prompt.char_to_token(loc + len(arg_temp + " <none>") - 1))
                    dec_mask_temp[arg_temp] = prompt_slots

                    # if prompt_slots["has_ans"] is True and self.training is False:
                    #     aaaa=1

                dec_role.append(dec_temp)
                dec_mask.append(dec_mask_temp)
            dec_prompt_ids = torch.tensor(dec_prompt_ids).to(self.config.device)
            dec_prompt_mask_ids = torch.tensor(dec_prompt_mask_ids).to(self.config.device)
            arg_joint_prompts = dec_role
            arg_joint_mask = dec_mask


        """
        Args:
            multi args post calculation
        """
        # if self.config.context_representation == 'decoder':  # encoder or decoder √
        #     context_outputs = self.model(
        #         enc_input_ids,
        #         attention_mask=enc_mask_ids,
        #         return_dict=True,
        #     )
        #     decoder_context = context_outputs.encoder_last_hidden_state
        #     context_outputs = context_outputs.last_hidden_state
        # else:
        #     context_outputs = self.model.encoder(
        #         enc_input_ids,
        #         attention_mask=enc_mask_ids,
        #     )
        #     context_outputs = context_outputs.last_hidden_state
        #     decoder_context = context_outputs
        #
        # decoder_prompt_outputs = self.model.decoder(
        #     input_ids=dec_prompt_ids,
        #     attention_mask=dec_prompt_mask_ids,
        #     encoder_hidden_states=decoder_context,
        #     encoder_attention_mask=enc_mask_ids,
        # )
        # prompt_outputs = decoder_prompt_outputs.last_hidden_state  # [bs, prompt_len, H]

        # if not self.training:
        #     outputs = self.model(
        #         input_ids=enc_input_ids,
        #         attention_mask=enc_mask_ids,
        #         return_dict=True,
        #         decoder_input_ids=None,
        #         decoder_attention_mask=None,
        #     )
        # else:

        # context_outputs = self.model(
        #         enc_input_ids,
        #         attention_mask=enc_mask_ids,
        #         return_dict=True,
        #     )
        # context_outputs = context_outputs.last_hidden_state
        # decoder_context = context_outputs
        # decoder_prompt_outputs = self.model.decoder(
        #     input_ids=dec_prompt_ids,
        #     attention_mask=dec_prompt_mask_ids,
        #     encoder_hidden_states=decoder_context,
        #     encoder_attention_mask=enc_mask_ids,
        # )
        # prompt_outputs = decoder_prompt_outputs.last_hidden_state

        outputs = self.model(
                input_ids=enc_input_ids,
                attention_mask=enc_mask_ids,
                return_dict=True,
                decoder_input_ids = dec_prompt_ids,
            decoder_attention_mask = dec_prompt_mask_ids,
            )
        context_outputs = outputs.encoder_last_hidden_state
        prompt_outputs = outputs.decoder_hidden_states[-1]


        # gen_logits = outputs.logits

        # context_outputs = self.model.encoder(
        #     enc_input_ids,
        #     attention_mask=enc_mask_ids,
        # )
        # context_outputs = context_outputs.last_hidden_state
        # # context_outputs = (context_outputs.hidden_states[0]+context_outputs.hidden_states[-1])/2
        #
        # prompt_outputs = self.model2.encoder(
        #     dec_prompt_ids,
        #     attention_mask=dec_prompt_mask_ids,
        # )
        # prompt_outputs = prompt_outputs.last_hidden_state
        # # decoder_prompt_outputs = (prompt_outputs.hidden_states[0]+prompt_outputs.hidden_states[-1])/2

        logit_lists = list()
        total_loss = 0.
        for i, (context_output, prompt_output, arg_joint_prompt, old_tok_to_new_tok_index, trigger, dec_prompt_id,
                enc_input_id, pt_mask) in \
                enumerate(
                    zip(context_outputs, prompt_outputs, arg_joint_prompts, old_tok_to_new_tok_indexs, event_trigger,
                        dec_prompt_ids, enc_input_ids, arg_joint_mask)):

            batch_loss = list()
            cnt = 0

            output = dict()

            i_temp = -1
            for arg_role in arg_joint_prompt.keys():

                i_temp += 1

                """
                "arg_role": {"tok_s": , "tok_e": }
                """
                prompt_slots = arg_joint_prompt[arg_role]
                mask_slots = pt_mask[arg_role]

                start_logits_list = list()
                end_logits_list = list()

                for (p_start, p_end, pt_s, pt_e) in zip(prompt_slots['tok_s'], prompt_slots['tok_e'], mask_slots['tok_s'], mask_slots['tok_e']):
                    prompt_query_sub = prompt_output[p_start:p_end]
                    # prompt_query_sub = torch.mean(prompt_query_sub, dim=0).unsqueeze(0)
                    prompt_query_sub = torch.mean(prompt_query_sub, dim=0)

                    trigger_sub = context_output[trigger[1][0]:trigger[1][1]]
                    trigger_sub = torch.mean(trigger_sub, dim=0)

                    # prompt_query_sub = torch.cat([prompt_query_sub,trigger_sub,torch.mul(prompt_query_sub,trigger_sub),prompt_query_sub-trigger_sub],dim=0)
                    # prompt_query_sub=self.w_p_start(prompt_query_sub).unsqueeze(0)

                    prompt_query_sub = torch.cat(
                        [prompt_query_sub, trigger_sub, torch.mul(prompt_query_sub, trigger_sub),
                         prompt_query_sub - trigger_sub], dim=0)

                    # prompt_query_sub = prompt_query_sub + trigger_sub

                    start_query = (self.w_context_start_t(prompt_query_sub)).unsqueeze(-1).unsqueeze(0)  # [1, H, 1]
                    end_query = (self.w_context_end_t(prompt_query_sub)).unsqueeze(-1).unsqueeze(0)  # [1, H, 1]
                    # start_query = (prompt_query_sub * self.w_context_start).unsqueeze(-1)  # [1, H, 1]
                    # end_query = (prompt_query_sub * self.w_context_end).unsqueeze(-1)  # [1, H, 1]
                    # position distributions  attention

                    # start_query = self.dropout(start_query)
                    # end_query = self.dropout(end_query)

                    # start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze().softmax(dim=0)
                    # end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze().softmax(dim=0)

                    # start_logits = torch.bmm(context_output.unsqueeze(0), start_query).squeeze() # 为什么不能在这里用softmax
                    # end_logits = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                    start_logits = F.log_softmax(torch.bmm(context_output.unsqueeze(0), start_query).squeeze(),
                                                 0)  # 为什么不能在这里用softmax
                    end_logits = F.log_softmax(torch.bmm(context_output.unsqueeze(0), end_query).squeeze(), 0)

                    # start_query = (trigger_sub * self.w_context_start).unsqueeze(-1)  # [1, H, 1]
                    # end_query = (trigger_sub * self.w_context_end).unsqueeze(-1)  # [1, H, 1]
                    # start_logits_t = torch.bmm(context_output.unsqueeze(0), start_query).squeeze()
                    # end_logits_t = torch.bmm(context_output.unsqueeze(0), end_query).squeeze()

                    # gen_logit = F.linear(prompt_output, self.model.shared.weight, bias=self.final_logits_bias)
                    # # gen_logit = F.linear(context_output, self.model.shared.weight, bias=self.final_logits_bias)
                    # # gen_logit = self.lm_head(prompt_output)
                    # # gen_probs = gen_logit[
                    # #     (dec_prompt_id == 50264).nonzero()[i_temp].item()]
                    # # gen_probs = gen_logit[
                    # #     (dec_prompt_id == 50264).nonzero()[i_temp].item()]
                    # gen_probs = gen_logit[p_start:p_end]
                    # gen_probs = torch.mean(gen_probs, dim=0)
                    # # gen_probs=torch.cat((gen_probs,torch.tensor([gen_probs[1],gen_probs[1]]).to(self.config.device)),0)
                    # # gen_probs=F.log_softmax(torch.index_select(gen_probs, 0, enc_input_id),0)
                    # real_probs = torch.index_select(gen_probs, 0, enc_input_id)
                    # gen_probs_s = F.log_softmax(real_probs * self.w_prompt_start,0)
                    # gen_probs_e = F.log_softmax(real_probs * self.w_prompt_end,0)
                    # # gen_probs_s = gen_probs_e = F.log_softmax(real_probs, 0)

                    # start_logits_list.append(start_logits)
                    # end_logits_list.append(end_logits)

                    # input_embeds = self.model.encoder.embed_tokens(
                    #     enc_input_id) * self.model.encoder.embed_scale
                    # # gen_logit = -self.__batch_dist__(input_embeds, prompt_output)
                    # gen_logit = torch.einsum('jk,lk->jl', prompt_output, input_embeds)
                    # gen_probs = gen_logit[p_start:p_end]
                    # gen_probs = torch.mean(gen_probs, dim=0)
                    # # # gen_probs = gen_logit[
                    # # #          (enc_input_id == 50264).nonzero()[i_temp].item()]
                    # gen_probs_s = F.log_softmax(gen_probs * self.w_prompt_start, 0)
                    # gen_probs_e = F.log_softmax(gen_probs * self.w_prompt_end, 0)
                    #
                    # # s_temp = start_logits * self.loss_ratio + gen_probs * (1-self.loss_ratio)
                    # # e_temp = end_logits * self.loss_ratio + gen_probs * (1-self.loss_ratio)
                    # # s_temp = start_logits * self.loss_ratio + gen_probs_s * (1 - self.loss_ratio)
                    # # e_temp = end_logits * self.loss_ratio + gen_probs_e * (1 - self.loss_ratio)
                    # s_temp = gen_probs_s
                    # e_temp = gen_probs_e
                    #
                    # # seen_lm_logits_s = torch.full_like(s_temp, fill_value=-1000).to(self.config.device)
                    # # seen_lm_logits_e = torch.full_like(s_temp, fill_value=-1000).to(self.config.device)
                    # # # for i in range((enc_input_id==0).nonzero()[1].item()):
                    # # temp_item = (enc_input_id==2).nonzero()[0].item()
                    # # seen_lm_logits_s[:temp_item] = s_temp[:temp_item]
                    # # seen_lm_logits_e[:temp_item] = e_temp[:temp_item]
                    # start_logits_list.append(s_temp)
                    # end_logits_list.append(e_temp)

                    # seen_lm = torch.full_like(gen_logit, fill_value=-1000).to(self.config.device)
                    # for j in range((enc_input_id == 2).nonzero()[1].item()):
                    #     seen_lm[:,enc_input_id[j]] = gen_logit[:,enc_input_id[j]]
                    # print(self.tokenizer.decode(torch.max(gen_logit, dim=1)[1]))
                    # print(self.tokenizer.decode(dec_prompt_id))

                    # start_logits_list.append(start_logits * self.loss_ratio + gen_probs * (1-self.loss_ratio))
                    # end_logits_list.append(end_logits * self.loss_ratio + gen_probs * (1-self.loss_ratio))

                    if self.loss_ratio != 1:
                        input_embeds = self.model.encoder.embed_tokens(
                            enc_input_id) * self.model.encoder.embed_scale
                        gen_logit = torch.einsum('jk,lk->jl', prompt_output, input_embeds)[pt_s: pt_e]
                        if mask_slots["has_ans"]:
                            lm_logits = self.convert_logit(gen_logit, enc_input_id)

                            assert self.training is True
                            # print("["+self.tokenizer.decode(torch.max(lm_logits, dim=1)[1])+"]"+"--"+np.array_str(torch.max(lm_logits, dim=1)[1].cpu().numpy()))
                            # print("["+self.tokenizer.decode(dec_prompt_id[pt_s:pt_e])+"]"+"--"+np.array_str(dec_prompt_id[pt_s:pt_e].cpu().numpy()))


                            batch_loss.append(self.loss_fct(lm_logits, dec_prompt_id[pt_s: pt_e]))
                        else:
                            assert len(gen_logit) == 1

                            lm_logits = self.convert_logit(gen_logit, enc_input_id)
                            # print("["+self.tokenizer.decode(torch.max(lm_logits, dim=1)[1])+"]"+"--"+np.array_str(torch.max(lm_logits, dim=1)[1].cpu().numpy()))
                            # print(", ".join(target_info[i][arg_role]["text"])+"--"+self.tokenizer.decode(dec_prompt_id[pt_s:pt_e]))

                            gen_probs_s = F.log_softmax(gen_logit[0] * self.w_prompt_start, 0)
                            gen_probs_e = F.log_softmax(gen_logit[0] * self.w_prompt_end, 0)

                            start_logits = start_logits * self.loss_ratio + gen_probs_s * (1 - self.loss_ratio)
                            end_logits = end_logits * self.loss_ratio + gen_probs_e * (1 - self.loss_ratio)


                    start_logits_list.append(start_logits)
                    end_logits_list.append(end_logits)

                # if self.loss_ratio != 1:
                #     input_embeds = self.model.encoder.embed_tokens(
                #         enc_input_id) * self.model.encoder.embed_scale
                #     gen_logit = torch.einsum('jk,lk->jl', prompt_output, input_embeds)[mask_slots['tok_s'][0]: mask_slots['tok_e'][0]]
                #     if mask_slots["has_ans"]:
                #         # lm_logits = torch.full((gen_logit.size(0), self.vocab_size), fill_value=-1000,
                #         #                        dtype=gen_logit.dtype).to(gen_logit.device)
                #         # index = enc_input_id.unsqueeze(dim=0).expand_as(gen_logit)
                #         # lm_logits.scatter_(dim=1, index=index, src=gen_logit)
                #         lm_logits=self.convert_logit(gen_logit, enc_input_id)
                #
                #         assert self.training is True
                #         # print(self.tokenizer.decode(torch.max(lm_logits, dim=1)[1]))
                #         # print(self.tokenizer.decode(dec_prompt_id[mask_slots['tok_s'][0]:mask_slots['tok_e'][0]]))
                #
                #         batch_loss.append(
                #             self.loss_fct(lm_logits, dec_prompt_id[mask_slots['tok_s'][0]: mask_slots['tok_e'][0]]))
                #     else:
                #         assert len(gen_logit) == 1
                #
                #         gen_probs_s = F.log_softmax(gen_logit[0] * self.w_prompt_start, 0)
                #         gen_probs_e = F.log_softmax(gen_logit[0] * self.w_prompt_end, 0)
                #
                #         lm_logits = self.convert_logit(gen_logit, enc_input_id)
                #         # print(self.tokenizer.decode(torch.max(lm_logits, dim=1)[1]))
                #         # print(target_info[i][arg_role]["text"])
                #
                #         for j in range(len(start_logits_list)):
                #             start_logits_list[j] = start_logits_list[j] * self.loss_ratio + gen_probs_s * (
                #                         1 - self.loss_ratio)
                #             end_logits_list[j] = end_logits_list[j] * self.loss_ratio + gen_probs_e * (
                #                         1 - self.loss_ratio)


                # if self.loss_ratio != 1:
                #     gen_logit = F.linear(prompt_output, self.model.shared.weight, bias=self.final_logits_bias)
                #     gen_logit = gen_logit[mask_slots['tok_s'][0]:mask_slots['tok_e'][0]]
                #     if mask_slots["has_ans"]:
                #         seen_lm = torch.full_like(gen_logit, fill_value=-1000).to(self.config.device)
                #         for j in range((enc_input_id == 2).nonzero()[0].item()):
                #             seen_lm[:, enc_input_id[j]] = gen_logit[:, enc_input_id[j]]
                #         print(self.tokenizer.decode(torch.max(seen_lm, dim=1)[1]))
                #         print(self.tokenizer.decode(dec_prompt_id[mask_slots['tok_s'][0]:mask_slots['tok_e'][0]]))
                #         if dec_prompt_id[mask_slots['tok_s'][0]:mask_slots['tok_e'][0]][0] == 50267:
                #             assert 1!=1
                #             batch_loss.append(self.loss_fct(seen_lm, dec_prompt_id[0: 1]))
                #         else:
                #             batch_loss.append(
                #                 self.loss_fct(seen_lm, dec_prompt_id[mask_slots['tok_s'][0]: mask_slots['tok_e'][0]]))
                #     else:
                #         real_probs = torch.index_select(gen_logit, 1, enc_input_id)
                #
                #         seen_lm = torch.full_like(gen_logit, fill_value=-1000).to(self.config.device)
                #         for j in range((enc_input_id == 2).nonzero()[0].item()):
                #             seen_lm[:, enc_input_id[j]] = gen_logit[:, enc_input_id[j]]
                #         print(self.tokenizer.decode(torch.max(seen_lm, dim=1)[1]))
                #         print(target_info[i][arg_role]["text"])
                #         # print(self.tokenizer.decode(dec_prompt_id[mask_slots['tok_s'][0]:mask_slots['tok_e'][0]]))
                #
                #         if self.num_mask <=1:
                #             assert len(real_probs) == 1
                #             for j in range(len(start_logits_list)):
                #                 start_logits_list[j]=start_logits_list[j] * self.loss_ratio + real_probs[0] * (1-self.loss_ratio)
                #                 end_logits_list[j] = end_logits_list[j] * self.loss_ratio + real_probs[0] * (1-self.loss_ratio)
                #                 # end_logits_list[j] = end_logits_list[j] * self.loss_ratio + real_probs[0] * (1-self.loss_ratio)
                #
                #         else:
                #             for j in range(len(start_logits_list)):
                #                 # start_logits_list[j]=start_logits_list[j] * self.loss_ratio + torch.sum(real_probs * torch.linspace(0, 1, self.num_mask).unsqueeze(1),dim=0)* (1-self.loss_ratio)
                #                 # end_logits_list[j] = end_logits_list[j] * self.loss_ratio + torch.sum(real_probs * torch.linspace(1, 0, self.num_mask).unsqueeze(1), dim=0)* (1-self.loss_ratio)
                #                 start_logits_list[j] = start_logits_list[j] * self.loss_ratio + real_probs[0] * (
                #                             1 - self.loss_ratio)
                #                 end_logits_list[j] = end_logits_list[j] * self.loss_ratio + real_probs[-1] * (
                #                             1 - self.loss_ratio)

                output[arg_role] = [start_logits_list, end_logits_list]

                if self.training:
                    # calculate loss
                    target = target_info[i][arg_role]  # "arg_role": {"text": ,"span_s": ,"span_e": }
                    predicted_spans = list()
                    for (start_logits, end_logits) in zip(start_logits_list, end_logits_list):
                        if self.config.matching_method_train == 'accurate':
                            predicted_spans.append(get_best_span(start_logits, end_logits, old_tok_to_new_tok_index,
                                                                 self.config.max_span_length))
                        elif self.config.matching_method_train == 'max':
                            predicted_spans.append(get_best_span_simple(start_logits, end_logits))
                        else:
                            raise AssertionError()

                    target_spans = [[s, e] for (s, e) in zip(target["span_s"], target["span_e"])]
                    if len(target_spans) < len(predicted_spans):
                        # need to consider whether to make more
                        pad_len = len(predicted_spans) - len(target_spans)
                        target_spans = target_spans + [[0, 0]] * pad_len
                        target["span_s"] = target["span_s"] + [0] * pad_len
                        target["span_e"] = target["span_e"] + [0] * pad_len

                    if self.config.bipartite:
                        idx_preds, idx_targets = hungarian_matcher(predicted_spans, target_spans)
                    else:
                        idx_preds = list(range(len(predicted_spans)))
                        idx_targets = list(range(len(target_spans)))
                        if len(idx_targets) > len(idx_preds):
                            idx_targets = idx_targets[0:len(idx_preds)]
                        idx_preds = torch.as_tensor(idx_preds, dtype=torch.int64)
                        idx_targets = torch.as_tensor(idx_targets, dtype=torch.int64)

                    cnt += len(idx_preds)
                    start_loss = self.loss_fct(torch.stack(start_logits_list)[idx_preds],
                                               torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets])
                    end_loss = self.loss_fct(torch.stack(end_logits_list)[idx_preds],
                                             torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets])

                    # start_prompt_loss = 0.
                    # end_prompt_loss = 0.
                    # if len(start_logits_prompt_list) == len(idx_preds):
                    #     start_prompt_loss = self.loss_fct(torch.stack(start_logits_prompt_list)[idx_preds],
                    #                                       torch.LongTensor(prompt_slots["tok_s"]).to(
                    #                                           self.config.device)[idx_targets])
                    #     end_prompt_loss = self.loss_fct(torch.stack(end_logits_prompt_list)[idx_preds],
                    #                                     torch.LongTensor(prompt_slots["tok_e"]).to(self.config.device)[
                    #                                         idx_targets])

                    # loss_token_energy = 0.
                    # for id in range(len(idx_preds)):
                    #     loss_hinge = HingeLoss()  # [self.tokentype_size-2, self.tokentype_size-1], self.tokentype_size-1==pad_token_label_id[0].item()
                    #     label_vec = self.label2vec(
                    #         torch.LongTensor(target["span_s"]).to(self.config.device)[idx_targets[id]], 500)
                    #     loss_token_hinge = loss_hinge(start_logits_list[idx_preds[id]],
                    #                                   label_vec)
                    #     # label_vec = self.label2vec(torch.LongTensor(prompt_slots["tok_s"]).to(self.config.device)[idx_targets[i]], 500)
                    #     _, pred_token = torch.max(torch.stack(start_logits_list)[idx_preds[id]], dim=0)
                    #     pred_vec = self.label2vec(pred_token, 500)
                    #     loss_token_energy += torch.max(torch.tensor([0, loss_token_hinge + self.token_energy_function(
                    #         context_output, label_vec) - self.token_energy_function(context_output, pred_vec)],
                    #                                                 dtype=torch.float))
                    #
                    #     loss_hinge = HingeLoss()  # [self.tokentype_size-2, self.tokentype_size-1], self.tokentype_size-1==pad_token_label_id[0].item()
                    #     label_vec = self.label2vec(
                    #         torch.LongTensor(target["span_e"]).to(self.config.device)[idx_targets[id]], 500)
                    #     loss_token_hinge = loss_hinge(end_logits_list[idx_preds[id]],
                    #                                   label_vec)
                    #     # label_vec = self.label2vec(
                    #     #     torch.LongTensor(prompt_slots["span_e"]).to(self.config.device)[idx_targets[i]], 500)
                    #     _, pred_token = torch.max(torch.stack(end_logits_list)[idx_preds[id]], dim=0)
                    #     pred_vec = self.label2vec(pred_token, 500)
                    #     loss_token_energy += torch.max(
                    #         torch.tensor([0, loss_token_hinge + self.token_energy_function(
                    #             context_output, label_vec) - self.token_energy_function(context_output, pred_vec)],
                    #                      dtype=torch.float))
                    #
                    # batch_loss.append(
                    #     ((start_loss + end_loss) / 2) * self.ratio_loss + loss_token_energy * (1 - self.ratio_loss))

                    if self.loss_ratio != 0:
                        batch_loss.append((start_loss + end_loss) / 2)

                    # batch_loss.append(((start_loss + end_loss) / 2) * self.ratio_loss + ((start_prompt_loss + end_prompt_loss)/2)*(1-self.ratio_loss) )
                    #
                    # if temp ==0:
                    #     batch_loss.append(((start_loss + end_loss) / 2))
                    # else:
                    #     batch_loss.append(((start_prompt_loss + end_prompt_loss) / 2))

            logit_lists.append(output)
            if self.training:  # inside batch mean loss
                total_loss = total_loss + torch.sum(torch.stack(batch_loss))# / cnt


        if self.training:
            return total_loss / len(context_outputs), logit_lists
        else:
            return [], logit_lists
