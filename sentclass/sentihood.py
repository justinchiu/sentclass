
import torch

import torchtext
from torchtext import data
from torchtext.data import Dataset, Field, Example, Iterator, TabularDataset

import io
import os
import json
import random

LOCATIONS = ["location1", "location2",]
ASPECTS = ["price", "safety", "transit-location", "general",]
ALL_ASPECTS = [
    "price",
    "safety",
    "transit-location",
    "general",
    "live",
    "quiet",
    "dining",
    "nightlife",
    "touristy",
    "shopping",
    "green-culture",
    "multicultural",
    "misc",
]
SENTIMENTS = ["positive", "negative", "none",]

#Y_shape = torch.Size([len(LOCATIONS), len(ALL_ASPECTS), len(SENTIMENTS)])
Y_shape = torch.Size([len(LOCATIONS), len(ASPECTS), len(SENTIMENTS)])

la2idx = {
    (l, a): i * len(LOCATIONS) + j
    for i, l in enumerate(LOCATIONS)
    for j, a in enumerate(ASPECTS)
    #for j, a in enumerate(ALL_ASPECTS)
}

def unzip(xs):
    return zip(*xs)

# opnions: [{"sentiment": , "aspect": , "location": }]
def get_opinions(opinions, text):
    labels = {
        (op["target_entity"].lower(), op["aspect"].lower()): op["sentiment"].lower()
        for op in opinions
    }
    locs = {
        l: any(l == x.lower() for x in text.split())
        for l in LOCATIONS
    }
    return list(unzip(
        (
            l,
            a,
            "none" if (l, a) not in labels else labels[(l,a)],
        )
        for l in LOCATIONS for a in ASPECTS
        if locs[l]
        #for l in LOCATIONS for a in ALL_ASPECTS
    ))

def get_all_opinions(opinions, text=None):
    labels = {
        (op["target_entity"].lower(), op["aspect"].lower()): op["sentiment"].lower()
        for op in opinions
    }
    return list(unzip(
        (
            l,
            a,
            "none" if (l, a) not in labels else labels[(l,a)],
        )
        for l in LOCATIONS for a in ASPECTS
        #for l in LOCATIONS for a in ALL_ASPECTS
    ))

def make_fields():
    SENTIMENT = Field(lower=True, is_target=True, unk_token=None, pad_token=None, batch_first=True)
    LOCATION = Field(lower=True, is_target=True, unk_token=None, pad_token=None, batch_first=True)
    ASPECT = Field(lower=True, is_target=True, unk_token=None, pad_token=None, batch_first=True)
    TEXT = Field(
        lower=False, include_lengths=True, is_target=True, batch_first=True, init_token="<bos>", eos_token="<eos>")
        #lower=True, include_lengths=True, init_token="<bos>", eos_token="<eos>", is_target=True)
    return TEXT, LOCATION, ASPECT, SENTIMENT

def build_vocab(f1, f2, f3, f4, d1, d2, d3):
    f1.build_vocab(d1, d2, d3)
    f2.build_vocab(d1, d2, d3)
    f3.build_vocab(d1, d2, d3)
    f4.build_vocab(d1, d2, d3)


class SentihoodExample(Example):

    @classmethod
    def fromJson(cls, data, text_field, location_field, aspect_field, sentiment_field):
        exs = []

        for x in json.load(data):
            ex = cls()

            # hmmmm get_opinions filters
            locations, aspects, sentiments = get_all_opinions(x["opinions"], x["text"])
            setattr(ex, "locations", location_field.preprocess(list(locations)))
            setattr(ex, "aspects", aspect_field.preprocess(list(aspects)))
            setattr(ex, "locations_text", text_field.preprocess(list(locations)))
            setattr(ex, "aspects_text", text_field.preprocess(list(aspects)))
            setattr(ex, "sentiments", sentiment_field.preprocess(list(sentiments)))
            setattr(ex, "text", text_field.preprocess(x["text"]))

            exs.append(ex)
        return exs, None, None, None

class SentihoodFlatExample(Example):

    @classmethod
    def fromJson(cls, data, text_field, location_field, aspect_field, sentiment_field):
        exs = []
        pos = []
        neg = []
        none = []

        posx = 0
        negx = 0
        nonex = 0
        tot = 0

        for x in json.load(data):
            #print(list(zip(*get_opinions(x["opinions"], x["text"]))))
            #print(x["text"])
            #import pdb; pdb.set_trace()
            for l, a, s in zip(*get_opinions(x["opinions"], x["text"])):
                ex = cls()

                setattr(ex, "locations", location_field.preprocess(l))
                setattr(ex, "aspects", aspect_field.preprocess(a))
                setattr(ex, "locations_text", text_field.preprocess(l))
                setattr(ex, "aspects_text", text_field.preprocess(a))
                setattr(ex, "sentiments", sentiment_field.preprocess(s))
                setattr(ex, "text", text_field.preprocess(x["text"]))

                exs.append(ex)

                if s == "negative":
                    negx += 1
                    neg.append(ex)
                elif s == "positive":
                    posx += 1
                    pos.append(ex)
                else:
                    nonex += 1
                    none.append(ex)
                tot += 1
        print(f"pos {posx} neg {negx} none {nonex}")
        print(tot)
        return exs, pos, neg, none


class SentihoodDataset(Dataset):

    @staticmethod
    def make_fields(text_field, location_field, aspect_field, sentiment_field):
        return [
            ("locations", location_field),
            ("aspects", aspect_field),
            ("locations_text", text_field),
            ("aspects_text", text_field),
            ("sentiments", sentiment_field),
            ("text", text_field),
        ]


    def __init__(
        self, path,
        text_field,
        location_field,
        aspect_field,
        sentiment_field,
        flat = False,
        **kwargs
    ):
        self.flat = flat

        # Sort by length of the text
        self.sort_key = lambda x: len(x.text)

        fields = self.make_fields(text_field, location_field, aspect_field, sentiment_field)

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            examples, pos, neg, none = (
                SentihoodExample.fromJson(f, text_field, location_field, aspect_field, sentiment_field)
                if not flat
                else SentihoodFlatExample.fromJson(f, text_field, location_field, aspect_field, sentiment_field)
            )
            self.pos = pos
            self.neg = neg
            self.none = none

            """
            if flat:
                examples, pos, neg, none = SentihoodFlatExample.fromJson(f, text_field, sentiment_field)
            else:
                examples = SentihoodExample.fromJson(f, text_field, sentiment_field)
                pos, neg, none = None, None, None
            """
        # unused
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(SentihoodDataset, self).__init__(examples, fields, **kwargs)


    @classmethod
    def splits(
        cls,
        text_field,
        location_field,
        aspect_field,
        sentiment_field,
        flat = False,
        path = None,
        root='.data',
        train='sentihood-train.json', validation='sentihood-dev.json', test='sentihood-test.json',
        **kwargs
    ):
        return super(SentihoodDataset, cls).splits(
            path = path,
            root = root,
            train = train,
            validation = validation,
            test = test,
            text_field = text_field,
            location_field = location_field,
            aspect_field = aspect_field,
            sentiment_field = sentiment_field,
            flat = flat,
            **kwargs
        )


    @classmethod
    def iters(
        cls,
        batch_size=32, device=0,
        root=".data", vectors=None,
        **kwargs
    ):
        pass



class RandomIterator(Iterator):
    """Defines an iterator that randomly selects subsets of equal size
    """

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(
                self.dataset.pos,
                self.dataset.neg,
                self.dataset.none,
                self.batch_size,
                self.sort_key, self.batch_size_fn,
                random_shuffler=self.random_shuffler,
                shuffle=self.shuffle,
                sort_within_batch=self.sort_within_batch)

def batch(data, batch_size, batch_size_fn=None):
    """Yield elements from data in chunks of batch_size."""
    if batch_size_fn is None:
        def batch_size_fn(new, count, sofar):
            return count
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = batch_size_fn(ex, len(minibatch), size_so_far)
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], batch_size_fn(ex, 1, 0)
    if minibatch:
        yield minibatch

def pool(pos, neg, none, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.
    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """
    if random_shuffler is None:
        random_shuffler = random.shuffle
    data3 = zip(random_shuffler(pos), random_shuffler(neg), random_shuffler(none))
    """
    data = [
        x
        for y in zip(random_shuffler(pos), random_shuffler(neg), random_shuffler(none))
        for x in y
    ]
    """
    for p in batch(data3, batch_size / 3, batch_size_fn):
        yield sorted([x for y in p for x in y], key=key)

if __name__ == "__main__":
    filepath = "/n/rush_lab/jc/code/sentclass/data"
    TEXT, LOCATION, ASPECT, SENTIMENT = make_fields()

    train, valid, test = SentihoodDataset.splits(
        TEXT, LOCATION, ASPECT, SENTIMENT, path=filepath
    )
    TEXT.build_vocab(train)
    SENTIMENT.build_vocab(train)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), batch_size=32, device=torch.device("cuda:0")
    )
    batch = next(iter(train_iter))
    import pdb; pdb.set_trace()
