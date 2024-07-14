# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
import torch
from fairseq import metrics, utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig


EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

@dataclass
class TranslationLabelConfig(TranslationConfig):
    lang_prefix_tok: Optional[str] = field(
        default=None,
    )
    infer_style: Optional[str] = field(
        default='t-enc',
    )

@register_task("translation_label", dataclass=TranslationLabelConfig)
class TranslationLabelTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationLabelConfig

    def __init__(self, cfg: TranslationLabelConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.lang_prefix_tok = cfg.lang_prefix_tok
        self.infer_style = cfg.infer_style


    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        if self.cfg.eval_bleu:
            tgt_token = criterion.get_tgt_token(sample)
            # logger.info("tgt_token = {}".format(tgt_token))
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model, tgt_token=tgt_token)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model, tgt_token = None):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None, tgt_token=tgt_token)

        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
        
    def inference_step(
            self, generator, models, sample, prefix_tokens=None, constraints=None, tgt_token = None,
    ):
        param_prefix_tokens = None
        if prefix_tokens is not None:
            param_prefix_tokens = prefix_tokens.clone()
        if self.lang_prefix_tok is not None:
            prefix_tokens = self.target_dictionary.index(self.lang_prefix_tok)
            assert prefix_tokens != self.target_dictionary.unk_index

            net_input = sample["net_input"]
            if "src_tokens" in net_input:
                src_tokens = net_input["src_tokens"]
            elif "source" in net_input:
                src_tokens = net_input["source"]
            else:
                raise Exception("expected src_tokens or source in net input")
            
            # bsz: total number of sentences in beam
            # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
            bsz, _ = src_tokens.size()[:2]
            prefix_tokens = torch.LongTensor([prefix_tokens]).unsqueeze(1)  # 1,1
            prefix_tokens = prefix_tokens.expand(bsz, -1)
            prefix_tokens = prefix_tokens.to(src_tokens.device)

            if self.infer_style != 't-enc':
                param_prefix_tokens = prefix_tokens
            # logger.info("infer_style = {}".format(self.infer_style))

        with torch.no_grad():
            if prefix_tokens is not None: # infer
                return generator.generate(models, sample, prefix_tokens=param_prefix_tokens, tgt_token=prefix_tokens)
            else: # train
                return generator.generate(models, sample, prefix_tokens=prefix_tokens, tgt_token=tgt_token)
