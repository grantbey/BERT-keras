import random
import numpy as np
from bert.data.vocab import TextEncoder
from typing import List, NamedTuple, Optional, Dict, Any


class TaskWeightScheduler:
    def __init__(self, active_in_pretrain = False, active_in_finetune = True,
                 pretrain_value = 1.0, finetune_value = 1.0):
        # type: (bool, bool, float, float) -> None
        self.active_in_pretrain = active_in_pretrain
        self.active_in_finetune = active_in_finetune
        self.pretrain_value = pretrain_value
        self.finetune_value = finetune_value

    def get(self, is_pretrain, step):
        # type: (bool, int)-> float
        if is_pretrain and self.active_in_pretrain:
            return self.pretrain_value
        if not is_pretrain and self.active_in_finetune:
            return self.finetune_value
        raise ValueError()


TaskMetadata = NamedTuple('TaskMetadata',
                          [('name', str), ('is_token_level', bool),
                           ('num_classes', int),
                           ('dropout', float),
                           ('weight_scheduler', Optional[TaskWeightScheduler])])


TokenTaskData = NamedTuple('TokenTaskData',
                           [('target', List[int]),
                            ('target_mask', List[bool])])


SentenceTaskData = NamedTuple('SentenceTaskData',
                              [('target', int),
                               ('target_index', int)])


TaskDataBatch = NamedTuple('TaskDataBatch',
                           [('target', np.array),
                            ('target_mask', np.array)])


Sentence = NamedTuple('Sentence',
                      [('tokens', List[int]),
                       ('padding_mask', List[bool]),
                       ('segments', Optional[List[int]]),
                       ('token_classification', Optional[Dict[str, TokenTaskData]]),
                       ('sentence_classification', Optional[Dict[str, SentenceTaskData]])])


SentenceBatch = NamedTuple('SentenceBatch',
                           [('tokens', np.array),
                            ('padding_mask', np.array),
                            ('segments', np.array),
                            ('token_classification', Dict[str, TaskDataBatch]),
                            ('sentence_classification', Dict[str, TaskDataBatch])])

def create_attention_mask(pad_mask, is_causal, batch_size = None,
                          length = None, bert_attention = False):
    # type: (Optional[np.array], bool, Optional[int], Optional[int], bool) -> np.array
    if pad_mask is not None:
        assert pad_mask.ndim == 2
        batch_size, length = pad_mask.shape
    if is_causal:
        b = np.cumsum(np.eye(length, dtype=np.float32), axis=0)
    else:
        b = np.ones((length, length), dtype=np.float32)
    b = np.reshape(b, [1, 1, length, length])
    b = np.repeat(b, batch_size, axis=0)  # B, 1, L, L
    if pad_mask is not None:
        _pad_mask = pad_mask[..., np.newaxis]
        _pad_mask = np.repeat(_pad_mask, length, 2)
        _pad_mask_t = np.transpose(_pad_mask, [0, 2, 1])
        if bert_attention:
            tmp = _pad_mask_t
        else:
            tmp = _pad_mask * _pad_mask_t
        tmp = tmp[:, np.newaxis, ...]
        if b is None:
            b = tmp.astype(np.float32)
        else:
            b = b * tmp
    return b


def _trim_seq(seq, length, from_end = True):
    # type: (Optional[List[Any]], int, bool) -> Optional[List[Any]]
    if seq is None:
        return None
    return seq[:length] if from_end else seq[-length:]


def _trim_sentence_target(task_dict, desired_len, orig_seq_len, from_end = True):
    # type: (Dict[str, SentenceTaskData], int, int, bool) -> Dict[str, SentenceTaskData]
    trimmed_task_dict = {}
    for k, v in task_dict.items():
        target_index = v.target_index
        if orig_seq_len > desired_len:
            if from_end and target_index > desired_len:
                target_index = -1
            if not from_end:
                target_index -= orig_seq_len - desired_len
        if target_index >= 0:
            trimmed_task_dict[k] = SentenceTaskData(v.target, target_index)
    return trimmed_task_dict


def _trim_sentence(sentence, length, from_end = True):
    # type: (Sentence, int, bool) -> Sentence
    return Sentence(_trim_seq(sentence.tokens, length, from_end),
                    _trim_seq(sentence.padding_mask, length, from_end),
                    _trim_seq(sentence.segments, length, from_end),
                    {k: TokenTaskData(_trim_seq(v.target, length, from_end),
                                      _trim_seq(v.target_mask, length, from_end)) for k, v in
                     sentence.token_classification.items()} if sentence.token_classification is not None else {},
                    _trim_sentence_target(sentence.sentence_classification, length, len(sentence.tokens),
                                          from_end) if sentence.sentence_classification is not None else {})


def check_sent_len(sentence, min_len, max_len, from_end = True):
    # type: (Sentence, Optional[int], bool, bool) -> Optional[Sentence]
    if min_len is not None and len(sentence.tokens) < min_len:
        return None
    if max_len is not None and len(sentence.tokens) > max_len:
        return _trim_sentence(sentence, max_len, from_end)
    return sentence


def msk_sentence(sentence, vocab_size, keep_prob, mask_prob, rand_prob):
    # type: (List[int], int, float, float, float) -> Sentence
    prediction_target = [0] * len(sentence)
    prediction_mask = [False] * len(sentence)
    new_sent = sentence.copy()
    for i in range(len(sentence)):
        probability = random.random()
        if probability > keep_prob:
            prediction_target[i] = sentence[i]
            prediction_mask[i] = True
            if probability < (mask_prob + keep_prob):
                new_sent[i] = vocab_size + TextEncoder.MSK_OFFSET
            elif probability < (mask_prob + rand_prob + keep_prob):
                new_sent[i] = random.randrange(vocab_size)
    return Sentence(new_sent, [True] * len(new_sent), None,
                    token_classification={'lm': TokenTaskData(prediction_target, prediction_mask)},
                    sentence_classification={})


def _pad_seq(seq, pad_token, pad_len, is_post_pad = True):
    # type: (List[int], int, int, bool) -> List[int]
    return (seq + [pad_token] * pad_len) if is_post_pad else ([pad_token] * pad_len + seq)


def pad(sentence, pad_id, max_len, is_post_pad = True):
    # type: (Sentence, int, int, bool) -> Sentence
    pad_len = max_len - len(sentence.tokens)
    if pad_len == 0:
        return sentence
    return Sentence(_pad_seq(sentence.tokens, pad_id, pad_len, is_post_pad),
                    _pad_seq(sentence.padding_mask, False, pad_len, is_post_pad),
                    _pad_seq(sentence.segments, 0, pad_len, is_post_pad),
                    {k: TokenTaskData(_pad_seq(v.target, 0, pad_len, is_post_pad),
                                      _pad_seq(v.target_mask, False, pad_len, is_post_pad)) for k, v in
                     sentence.token_classification.items()} if sentence.token_classification is not None else {},
                    {k: SentenceTaskData(v.target, v.target_index + (0 if is_post_pad else pad_len)) for k, v in
                     sentence.sentence_classification.items()} if sentence.sentence_classification is not None else {})


def generate_pos_ids(batch_size, max_len):
    # type: (int, int) -> np.array
    return np.repeat(np.arange(max_len, dtype=np.int32).reshape(1, -1), batch_size, 0)
