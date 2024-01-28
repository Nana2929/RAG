import copy
from collections import Counter, defaultdict

import numpy as np
import scipy
import torch
import torch.nn as nn
from pepler.modeling_pepler_buildingblocks import (
    MLP,
    ContinuousModelOutput,
    OverallMLP,
    RecModelOutput,
    WeightedMLP,
)
from transformers import GPT2LMHeadModel

# =============================================================
# Continous Prompt Learning w/ recommendation as regularization
# =============================================================



class UIAPromptWithReg:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,
                        nuser, nitem, naspect, use_mlp=True, *args,
                        **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path)
        model.init_prompt(nuser, nitem, naspect, use_mlp)

        return model

    def init_prompt(self, nuser, nitem, naspect, use_mlp=True):
        # update all kwargs to self
        self.src_len = 3
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.aspect_embeddings = nn.Embedding(naspect, emsize)

        if use_mlp:
            self.rec = MLP(emsize)
        else:
            self.rec = WeightedMLP(emsize)

        self.overall_rec = OverallMLP(naspect)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.aspect_embeddings.weight.data.uniform_(-initrange, initrange)

    def init_pretrained_prompts(self, userfeats, itemfeats, aspectfeats,
                                pad_token_id,
                                device):
        """We can set them as the vectors from embedding layer
        or the vectors from the last layer of the transformer. # v1
        # v1:
        For embedding layer, see https://github.com/huggingface/transformers/issues/1458.
        https://github.com/mkshing/Prompt-Tuning/blob/master/model.py

        init_prompt_value = self.transformer.wte[inputs]
        """
        upw, ipw, apw = [], [], []
        # !! https://github.com/huggingface/transformers/issues/1458#issuecomment-1564253345
        for userf in userfeats: # (nuser, prompt_tokens_maxlen)
            # erase the <pad> in userf tensor
            userf = userf[userf != pad_token_id]
            weights = self.transformer.wte.weight[userf, :] # (emsize, src_len)
            weights = torch.mean(weights, dim=0).squeeze()
            upw.append(weights.detach().cpu().numpy()) # (emsize,)


        for itemf in itemfeats:
            itemf = itemf[itemf != pad_token_id]
            weights = self.transformer.wte.weight[itemf, :]
            weights = torch.mean(weights, dim=0).squeeze()
            ipw.append(weights.detach().cpu().numpy())
        for aspectf in aspectfeats:
            aspectf = aspectf[aspectf != pad_token_id]
            weights = self.transformer.wte.weight[aspectf, :]
            weights = torch.mean(weights, dim=0).squeeze()
            apw.append(weights.detach().cpu().numpy())

        upw = torch.tensor(upw).to(device)
        ipw = torch.tensor(ipw).to(device)
        apw = torch.tensor(apw).to(device)
        # see device
        print(f"user soft-prompt shape: {upw.shape}, item soft-prompt shape: {ipw.shape}, aspect soft-prompt shape: {apw.shape}")
        self.__set_user_embeddings(upw)
        self.__set_item_embeddings(ipw)
        self.__set_aspect_embeddings(apw)

    def __set_aspect_embeddings(self, aspect_weights):
        self.aspect_embeddings.weight.data = aspect_weights

    def __set_user_embeddings(self, user_weights):
        self.user_embeddings.weight.data = user_weights

    def __set_item_embeddings(self, item_weights):
        self.item_embeddings.weight.data = item_weights

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_plm_only(self):
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, user, item, aspect,
                text, mask,
                aspect_score = None,
                overall_rating: int=None, rating_prediction=True, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        a_src = self.aspect_embeddings(aspect)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), a_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if rating_prediction:
            aspect_rating = self.rec(u_src, i_src, a_src).squeeze()
            # (batch_size,)
        else:
            aspect_rating = None

        if aspect_score is not None:
            overall_rating = self.overall_rec(aspect_score).squeeze() # (batch_size,)
        outputs = super().forward(inputs_embeds=src)
        if mask is None:
            # auto-regressive generation
            # return super().forward(inputs_embeds=src), aspect_rating
            outputs = super().forward(inputs_embeds=src)
            return RecModelOutput(
                logits=outputs.logits,
                aspect_rating=aspect_rating,
                overall_rating=overall_rating,
                outputs=outputs
            )
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # (batch_size, src_len)
            # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            outputs = super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)
            return RecModelOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                aspect_rating=aspect_rating,
                overall_rating=overall_rating,
                outputs=outputs
            )


class RecReg(UIAPromptWithReg, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


# =============================================================
# Continous Prompt Learning w/o recommendation as regularization
# =============================================================
class UIAPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, naspect, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem, naspect)
        return model

    def init_prompt(self, nuser, nitem, naspect):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        self.aspect_embeddings = nn.Embedding(naspect, emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.aspect_embeddings.weight.data.uniform_(-initrange, initrange)


    def set_aspect_embeddings(self, aspect_weights):
        self.aspect_embeddings.weight.data = aspect_weights


    def forward(self, user, item, text, aspect, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        a_src = self.aspect_embeddings(aspect)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1),
                        a_src.unsqueeze(1),
                        w_src], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return ContinuousModelOutput(
                logits=super().forward(inputs_embeds=src).logits,
            )
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size,src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)
            outputs = super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

            return ContinuousModelOutput(
                loss=outputs.loss,
                logits=outputs.logits
            )

class ContinuousPromptLearning(UIAPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

# =============================================================
# Discrete Prompt Learning
    # no recommendation as regularization by default
    # !! have not revised the output type yet !!
# =============================================================

class FeaturePrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, context, explanation, exp_mask, ignore_index=-100):
        device = context.device
        text = torch.cat([context, explanation], 1)  # (batch_size, total_len)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)

        if exp_mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones_like(context, dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, exp_mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full_like(context, ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(exp_mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class DiscretePromptLearning(FeaturePrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)

