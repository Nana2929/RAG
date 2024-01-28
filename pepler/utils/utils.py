import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result

    return wrapper


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path):
    return pd.read_csv(path)


def identity_tokenizer(text: list[str]):
    assert isinstance(text, list)
    return text


def create_run_dir(checkpoint_dir):
    # traverse the checkpoint_dir to find the largest run_dir
    # run_{max}
    dirs = os.listdir(checkpoint_dir)
    max_run_index = 0
    for d in dirs:  # run_{index}
        if d.startswith("run_"):
            index = int(d.split("_")[1])
            max_run_index = max(max_run_index, index)
    run_dir = os.path.join(checkpoint_dir, "run_{}".format(max_run_index + 1))
    os.mkdir(run_dir)
    return run_dir


def load_model(fp: os.PathLike):
    with open(fp, "rb") as f:
        model = torch.load(f)
    model.eval()
    return model



class FfidfStore:
    def __init__(self, cache_dir: os.PathLike):
        self.cache_dir = Path(cache_dir)
        assert self.cache_dir.exists(), f"{self.cache_dir} does not exist"
        print(f"Loading precomputed ffidf values from {self.cache_dir}")
        self.user_ffidf = load_pickle(self.cache_dir / "user_ffidf.pickle")
        self.item_ffidf = load_pickle(self.cache_dir / "item_ffidf.pickle")
        self.user_feat_names = load_pickle(self.cache_dir / "user_feat_names.pickle")
        self.item_feat_names = load_pickle(self.cache_dir / "item_feat_names.pickle")
        self.idx2user = load_pickle(self.cache_dir / "idx2user.pickle")
        self.idx2item = load_pickle(self.cache_dir / "idx2item.pickle")

        # revert the idx2user and idx2item
        self.user2idx = {v: k for k, v in self.idx2user.items()}
        self.item2idx = {v: k for k, v in self.idx2item.items()}
        print(f"Loaded {len(self.user2idx)} users and {len(self.item2idx)} items.")

        print(f"Loading completed.")

    def _get_ffidf(
        self,
        id: str,
        feat_names: list[str],
        ffidf: scipy.sparse.csr_matrix,
        topk=10,
        return_score=False,
    ):
        feat_index = ffidf[id, :].nonzero()[1]
        # get feat_names
        feat_names = [feat_names[i] for i in feat_index]
        ffidf_scores = zip(feat_names, [ffidf[id, x] for x in feat_index])
        # sort by score (descending)
        ffidf_scores = sorted(ffidf_scores, key=lambda x: -x[1])
        if return_score:
            return ffidf_scores[:topk]
        else:
            return [f for (f, _) in ffidf_scores[:topk]]

    def get_user_ffidf(self, user_id: str, topk: int, return_score: bool = False):
        if topk < 0:
            topk = len(self.user_feat_names)
        return self._get_ffidf(
            self.user2idx[user_id],
            self.user_feat_names,
            self.user_ffidf,
            topk,
            return_score,
        )

    def get_item_ffidf(self, item_id: str, topk: int, return_score: bool = False):
        if topk < 0:
            topk = len(self.item_feat_names)
        return self._get_ffidf(
            self.item2idx[item_id],
            self.item_feat_names,
            self.item_ffidf,
            topk,
            return_score,
        )


class ReviewHistory:
    """
    a class for fetching review history
    at the cost of ram
    """

    def __init__(self, data, valid_data=None, test_data=None, logger=None):
        self.text_table = []
        self.feature_table = []
        self.user2_texttableids = defaultdict(list)
        self.item2_texttableids = defaultdict(list)
        self.ui2_texttableid = dict()
        self.user2item = defaultdict(list)
        self.item2user = defaultdict(list)
        self.logger = logger

        cur_texttable_id = 0
        self.text_table.append("")  # stub empty string
        cur_texttable_id += 1
        ratings = []
        cur_featuretable_id = 0
        self.feature_table.append("")  # offset
        cur_featuretable_id += 1
        for sample in data:
            self.text_table.append(sample["text"])
            self.feature_table.append(sample["feature"])
            self.user2_texttableids[sample["user"]].append(cur_texttable_id)
            self.item2_texttableids[sample["item"]].append(cur_texttable_id)
            self.ui2_texttableid[(sample["user"], sample["item"])] = cur_texttable_id
            self.user2item[sample["user"]].append(sample["item"])
            self.item2user[sample["item"]].append(sample["user"])
            cur_texttable_id += 1
            cur_featuretable_id += 1
            ratings.append(sample["rating"])

        self._mean_rating = np.average(ratings)
        ratings.insert(0, self._mean_rating)  # offset for the stub review
        self.rating_table = np.array(ratings)

        # if valid and test data is not None, extend text table to
        # keep those reviews as well.
        # but do not keep those in history.
        if valid_data is not None:
            for sample in valid_data:
                self.text_table.append(sample["text"])
                self.ui2_texttableid[
                    (sample["user"], sample["item"])
                ] = cur_texttable_id
                self.user2item[sample["user"]].append(sample["item"])
                self.item2user[sample["item"]].append(sample["user"])
                cur_texttable_id += 1

        if test_data is not None:
            for sample in test_data:
                self.text_table.append(sample["text"])
                self.ui2_texttableid[
                    (sample["user"], sample["item"])
                ] = cur_texttable_id
                self.user2item[sample["user"]].append(sample["item"])
                self.item2user[sample["item"]].append(sample["user"])
                cur_texttable_id += 1

        self.text_table = np.array(self.text_table, dtype=object)
        self.feature_table = np.array(self.feature_table, dtype=object)

    def nuser(self):
        return len(self.user2item)

    def nitem(self):
        return len(self.item2user)

    def nreview(self):
        return len(self.text_table)

    def mean_rating(self):
        return self._mean_rating

    def load_embedded_text_table(self, table_dir):
        self.raw_text_table = np.array([t for t in self.text_table])
        with open(table_dir, "rb") as f:
            self.text_table = np.load(f)

    def build_embedded_text_table(
        self, sentence_transformer, device, maybe_load_from=None
    ):
        if maybe_load_from is not None:
            self.logger.info("trying to load reviews")
            try:
                self.load_embedded_text_table(maybe_load_from)
                return self
            except Exception as e:
                self.logger.info(e)
                self.logger.info("loading failed")

        self.raw_text_table = np.array([t for t in self.text_table])
        self.text_table = sentence_transformer.encode(
            self.text_table.tolist(),
            convert_to_numpy=True,
            device=device,
            show_progress_bar=True,
        )
        if maybe_load_from is not None:
            np.save(open(maybe_load_from, "wb"), self.text_table)
        return self

    def load_embedded_text_table(self, table_path):
        self.logger.info("loading embedded reviews")
        self.raw_text_table = np.array([t for t in self.text_table])
        self.text_table = np.load(table_path)
        return self

    def get_user_history(
        self, user, hide_item=None, truncation=True, return_embedding=True
    ):
        if hide_item != None and (user, hide_item) in self.ui2_texttableid:
            history = [
                i
                for i in self.user2_texttableids[user]
                if i != self.ui2_texttableid[(user, hide_item)]
            ]
        else:
            history = [i for i in self.user2_texttableids[user]]

        if not return_embedding:
            return history
        if truncation:
            return np.mean(self.text_table[history], axis=-1)
        return self.text_table[history]

    def get_item_history(
        self, item, hide_user=None, truncation=True, return_embedding=True
    ):
        if hide_user != None and (hide_user, item) in self.ui2_texttableid:
            history = [
                i
                for i in self.item2_texttableids[item]
                if i != self.ui2_texttableid[(hide_user, item)]
            ]
        else:
            history = [i for i in self.item2_texttableids[item]]

        if not return_embedding:
            return history
        if truncation:
            return np.mean(self.text_table[history], axis=-1)
        return self.text_table[history]

    def get_ui_review(self, user, item, return_embedding=True):
        if return_embedding:
            return self.text_table[self.ui2_texttableid[(user, item)]]
        return self.ui2_texttableid[(user, item)]
