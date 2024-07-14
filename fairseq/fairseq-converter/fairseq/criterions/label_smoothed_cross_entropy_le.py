import math

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig
)
from omegaconf import II
from fairseq import metrics, utils
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)

@dataclass
class LabelSmoothedCrossEntropyCriterionWithKLConfig(
    LabelSmoothedCrossEntropyCriterionConfig
):
    label_style: str = field(
        default="t-enc",
    )

@register_criterion(
    "label_smoothed_cross_entropy_le", 
    dataclass=LabelSmoothedCrossEntropyCriterionWithKLConfig
)
class LabelSmoothedCrossEntropyCriterionLE(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 label_style="t-enc",
                 ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.label_style = label_style
    
    def get_tgt_token(self, sample):

        if self.label_style == "t-enc":
            src_tokens = sample["net_input"]["src_tokens"]

            mask = src_tokens.eq(self.padding_idx)
            indices = mask.sum(dim=1, keepdim=True)
            tgt_id = src_tokens.gather(1, indices)
        else:
            target = sample["target"]
            tgt_id = target[:, :1]            

        return tgt_id
     
    def forward(self, model, sample, reduce=True):
        # net_output = model(**sample["net_input"])
        # [seqlen, batch, dim]
        tgt_token = self.get_tgt_token(sample)
        encoder_out = model.encoder.forward(src_tokens = sample["net_input"]["src_tokens"], 
                                            src_lengths = sample["net_input"]["src_lengths"], 
                                            return_all_hiddens = True,
                                            tgt_token = tgt_token)

        net_output = model.decoder.forward(prev_output_tokens = sample["net_input"]["prev_output_tokens"],
                                           encoder_out = encoder_out,
                                           src_lengths = sample["net_input"]["src_lengths"])

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]

        all_loss = loss 
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return all_loss, sample_size, logging_output
