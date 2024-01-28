import os
import pickle
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset, WeightedRandomSampler

from .utils import AspectDataBatch, EntityDictionary, FfidfStore

"""
{'review_id': '110922565534353024196_604a5d09d863393130a413fa',
'user': '110922565534353024196',
'item': '604a5d09d863393130a413fa',
'rating': 4,
'template': ('ramen', 'good', 'good ramen', 'positive', 'Japanese'),
'triplets':
    [
        ('ramen', 'good', 'good ramen', 'positive', 'Japanese'),
        ('broth', 'flavorful', 'flavorful broth', 'positive', 'ambience')
    ],
'text': 'Very good ramen with flavorful broth.',
'tokens': ['Very', 'good', 'ramen', 'with', 'flavorful', 'broth', '.'],
'pics': ['AF1QipPaM_gz9dcOcO4HGD7OrPIwZqhX_be1HE1PIptm']}

"""


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end-start} seconds")
        return result

    return wrapper


class AspectDataInitializer:
    def __init__(
        self,
        data_path,
        aspect_path,
        index_dir,
        tokenizer=None,
        seq_len=None,
        *args,
        **kwargs,
    ):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        # self.max_rating = float("-inf")
        # self.min_rating = float("inf")

        # aspect2idx is inverted index of aspect list
        self.aspect_list = []
        self.aspect2idx = {}
        self.feature_set = set()

        self.data_path = data_path
        self.aspect_path = aspect_path
        self.initialize(data_path)
        self.initialize_aspect(aspect_path)

        # self.tokenizer = tokenizer
        self.seq_len = seq_len
        (
            self.train,
            self.valid,
            self.test,
            self.user2feature,
            self.item2feature,
        ) = self.load_data(self.data_path, index_dir)

    def initialize_aspect(self, aspect_path):
        assert os.path.exists(aspect_path)
        # csv file
        aspects = pd.read_csv(aspect_path)
        self.aspect_list = aspects["category"].tolist()
        self.aspect2idx = {aspect: idx for idx, aspect in enumerate(self.aspect_list)}

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, "rb"))

        for review in reviews:
            self.user_dict.add_entity(review["user"])
            self.item_dict.add_entity(review["item"])
            # overall_rating = review["overall_rating"]

            feat = review["template"][0]
            if isinstance(feat, list):
                self.feature_set.update(feat)
            else:  # isinstance(feat, str):
                self.feature_set.add(feat)

            # if self.max_rating < aspect_rating:
            #     self.max_rating = aspect_rating
            # if self.min_rating > aspect_rating:
            #     self.min_rating = aspect_rating
        print("feature set size: {}".format(len(self.feature_set)))

    def load_data(self, data_path: os.PathLike, index_dir: os.PathLike):
        """_summary_

        Parameters
        ----------
        data_path : os.PathLike
            path to the data file (reviews.pickle)
        index_dir : os.PathLike
            path to the index directory (k-fold cross validation)
        Returns
        -------
        train : list, list of reviews for training
        valid : list, list of reviews for validation
        test : list, list of reviews for testing
        user2feature : dict, user to feature mapping
        item2feature : dict, item to feature mapping
        """
        reviews = pickle.load(open(data_path, "rb"))
        _data = defaultdict(list)
        aspect_scores = defaultdict(lambda: np.zeros(len(self.aspect_list)))
        data = [[] for _ in range(len(reviews))]

        for idx, review in enumerate(reviews):
            feat, adj, text, sco, category = review["template"]
            category_idx = self.aspect2idx[category]
            # tokens = self.tokenizer(text)["input_ids"]
            # text = self.tokenizer.decode(
            #     tokens[: self.seq_len]
            # )  # keep seq_len tokens at most
            ui_id = f'{review["user"]}_{review["item"]}'
            _data[ui_id].append(
                {
                    "user": self.user_dict.entity2idx[review["user"]],
                    "item": self.item_dict.entity2idx[review["item"]],
                    "triplets": review["triplets"],
                    "rating": sco,  # see enrich_rec_dataset/convert_{}_nete.py:merge_triplets()
                    "text": text,
                    "feature": feat,
                    "overall_rating": review["rating"],
                    "category": category_idx,
                    "category_name": category,
                    "idx": idx,
                }
            )
            aspect_scores[ui_id][category_idx] = sco
        # flattening the review segments
        for ui_id, segments in _data.items():
            for seg in segments:
                seg["aspect_score"] = aspect_scores[ui_id]
        # flatten the data
        for ui_id, segments in _data.items():
            for seg in segments:
                # seg["idx"] records the index of the review in the original data
                data[seg["idx"]] = seg
        del _data
        # load train, test, valid index
        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review["user"]
            i = review["item"]
            f = review["feature"]
            if u in user2feature:
                user2feature[u].append(f) if isinstance(f, str) else user2feature[
                    u
                ].extend(f)
            else:
                user2feature[u] = [f] if isinstance(f, str) else f
            if i in item2feature:
                item2feature[i].append(f) if isinstance(f, str) else item2feature[
                    i
                ].extend(f)
            else:
                item2feature[i] = [f] if isinstance(f, str) else f
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, "train.index"), "r") as f:
            train_index = [int(x) for x in f.readline().split(" ")]
        with open(os.path.join(index_dir, "validation.index"), "r") as f:
            valid_index = [int(x) for x in f.readline().split(" ")]
        with open(os.path.join(index_dir, "test.index"), "r") as f:
            test_index = [int(x) for x in f.readline().split(" ")]
        return train_index, valid_index, test_index


class AspectDataWithPromptInitializer(AspectDataInitializer):
    def __init__(
        self,
        data_path,
        aspect_path,
        index_dir,
        tokenizer,
        seq_len,
        ffidf_cache_dir,
        ffidf_topk,
        prompt_tokens_maxlen,
        **kwargs,
    ):
        super().__init__(
            data_path, aspect_path, index_dir, tokenizer, seq_len, **kwargs
        )
        self.ffidf_store = FfidfStore(ffidf_cache_dir)
        self.ffidf_topk = ffidf_topk
        self.prompt_tokens_len = prompt_tokens_maxlen
        # how many tokens (to be averaged) for 1 prompt token
        print("Loading prompts...")
        self.uprompts, self.iprompts, self.aprompts = self.load_prompts()

    @timer
    def load_prompts(self):
        uprompt, iprompt, aprompt = [], [], []
        for _, user in enumerate(self.user_dict.idx2entity):
            up = self.ffidf_store.get_user_ffidf(
                user, self.ffidf_topk, return_score=False
            )
            uprompt.append(" ".join(up))
        for _, item in enumerate(self.item_dict.idx2entity):
            ip = self.ffidf_store.get_item_ffidf(
                item, self.ffidf_topk, return_score=False
            )
            iprompt.append(" ".join(ip))
        for _, aspect in enumerate(self.aspect_list):
            aprompt.append(aspect)

        encoded_up = self.tokenizer(
            uprompt,
            truncation=True,
            padding=True,
            max_length=self.prompt_tokens_len,
            return_tensors="pt",
        )
        encoded_ip = self.tokenizer(
            iprompt,
            truncation=True,
            padding=True,
            max_length=self.prompt_tokens_len,
            return_tensors="pt",
        )
        encoded_ap = self.tokenizer(
            aprompt,
            truncation=True,
            padding=True,
            max_length=self.prompt_tokens_len,
            return_tensors="pt",
        )
        encoded_up = encoded_up["input_ids"].contiguous()  # (nuser, seq_len)
        encoded_ip = encoded_ip["input_ids"].contiguous()  # (nitem, seq_len)
        encoded_ap = encoded_ap["input_ids"].contiguous()  # (naspect, seq_len)
        return encoded_up, encoded_ip, encoded_ap


from dataclasses import dataclass


@dataclass
class UserItemData:
    user: int
    item: int
    rating: float
    segments: list[dict]

    def __iter__(self):
        return iter(self.segments)

    def __getitem__(self, index):
        return self.segments[index]


class UserItemDataset(Dataset):
    """
    This dataset is not used for training, but for evaluation only.
    The batch size is dynamic =  how many aspect categories the user-item pair has.
    """

    def __init__(self, data, tokenizer, bos, eos, *args, **kwargs):
        self.bos = bos
        self.eos = eos
        self.tokenizer = tokenizer

        user_item_reviews = defaultdict(list)
        for x in data:
            user, item = x["user"], x["item"]
            user_item_reviews[(user, item)].append(x)
        self.data = [v for k, v in user_item_reviews.items()]

    def __repr__(self):
        return "<UserItemDataset> Note that all user-item reviews count as one data"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ds = self.data[index]
        u = ds[0]["user"]
        i = ds[0]["item"]
        r = ds[0]["overall_rating"]
        return UserItemData(u, i, r, ds)

    def collate_fn(self, uidata):
        # check https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
        u, i, a, r, t = [], [], [], [], []
        category_names = []
        aspect_scores, overall_ratings = [], []
        for segment in uidata:
            u.append(uidata.user)
            i.append(uidata.item)
            a.append(segment["category"])
            r.append(segment["rating"])
            category_names.append(segment["category_name"])
            t.append("{} {} {}".format(self.bos, segment["text"], self.eos))
            aspect_scores.append(segment["aspect_score"])
            overall_ratings.append(uidata.rating)

        user = torch.tensor(u, dtype=torch.int64).contiguous()
        item = torch.tensor(i, dtype=torch.int64).contiguous()
        aspect = torch.tensor(a, dtype=torch.int64).contiguous()
        rating = torch.tensor(r, dtype=torch.float).contiguous()
        overall_rating = torch.tensor(overall_ratings, dtype=torch.float).contiguous()
        aspect_scores = np.array(aspect_scores)
        encoded_inputs = self.tokenizer(t, padding=True, return_tensors="pt")
        seq = encoded_inputs["input_ids"].contiguous()
        mask = encoded_inputs["attention_mask"].contiguous()

        return AspectDataBatch(
            user=user,
            item=item,
            aspect=aspect,
            rating=rating,
            overall_rating=overall_rating,
            aspect_score=torch.tensor(aspect_scores, dtype=torch.float).contiguous(),
            seq=seq,
            mask=mask,
            category_name=category_names,
        )


class AspectDataset(Dataset):
    def __init__(self, data, tokenizer, bos, eos, *args, **kwargs):
        self.data = data
        u, i, a, r, t, category, feature = [], [], [], [], [], [], []
        aspect_scores, overall_ratings = [], []
        # pending: make (offset_mapping = (start, end)) for each user-item pair (may have multiple aspect triplets)
        # because shuffling will change the order of user-item pairs
        category_names = []
        for x in data:
            u.append(x["user"])
            i.append(x["item"])
            a.append(x["category"])
            r.append(x["rating"])
            t.append("{} {} {}".format(bos, x["text"], eos))

            category_names.append(x["category_name"])
            category.append(x["category"])
            feature.append(x["feature"])
            aspect_scores.append(x["aspect_score"])
            overall_ratings.append(x["overall_rating"])

        encoded_inputs = tokenizer(t, padding=True, return_tensors="pt")
        # * uia prompt tuning (uia to aspect text span prediction) *
        self.seq = encoded_inputs["input_ids"].contiguous()
        self.mask = encoded_inputs["attention_mask"].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.aspect = torch.tensor(a, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()  # aspect rating
        self.overall_rating = torch.tensor(
            overall_ratings, dtype=torch.float
        ).contiguous()
        aspect_scores = np.array(
            aspect_scores
        )  # exploiting the fast conversion from np to tensor
        aspect_score = torch.tensor(aspect_scores, dtype=torch.float).contiguous()
        # 1. normalize to [-1,1] and 2. keep the sign (positive, negative)
        # https://stackoverflow.com/questions/69352980/normalize-an-array-of-floats-into-a-certain-range-with-keeping-sign-in-python
        print(
            "Original aspect score with score range: [{}, {}]".format(
                aspect_score.min(), aspect_score.max()
            )
        )

        aspect_score = aspect_score / abs(aspect_score).max()
        print(
            "Normalizing aspect score vectors. Aspect score range: [{}, {}]".format(
                aspect_score.min(), aspect_score.max()
            )
        )
        self.aspect_score = aspect_score
        self.min_aspect_score = -1
        self.max_aspect_score = 1

        self.category = category
        self.category_names = category_names
        self.feature = feature

    def get_weighted_sampler(self):
        # https://www.kaggle.com/code/mineshjethva/pytorch-weighted-sampler-for-imbalance-classifier
        cnt = Counter(self.category)
        class_sample_count = np.array([cnt[i] for i in range(len(cnt))])
        weights = 1.0 / class_sample_count
        samples_weights = weights[self.category]
        sampler = WeightedRandomSampler(
            samples_weights, len(samples_weights), replacement=True
        )
        return sampler

    def collate_fn(self, batch):
        # check https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate
        users, items, aspects, ratings, overall_ratings, aspect_scores = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        seqs, masks = [], []
        for u, i, a, r, t, m, o, s in batch:
            users.append(u)
            items.append(i)
            aspects.append(a)
            ratings.append(r)
            overall_ratings.append(o)
            aspect_scores.append(s)
            seqs.append(t)
            masks.append(m)
        return AspectDataBatch(
            user=torch.stack(users),
            item=torch.stack(items),
            aspect=torch.stack(aspects),
            rating=torch.stack(ratings),
            overall_rating=torch.stack(overall_ratings),
            aspect_score=torch.stack(aspect_scores),
            seq=torch.stack(seqs),
            mask=torch.stack(masks),
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        aspect = self.aspect[index]
        rating = self.rating[index]
        overall_rating = self.overall_rating[index]
        aspect_score = self.aspect_score[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        return user, item, aspect, rating, seq, mask, overall_rating, aspect_score
