
import torch

import torchtext
from torchtext import data
from torchtext.data import Dataset, Field, Example, Iterator, TabularDataset

import io
import os
import json

LOCATIONS = ["location1", "location2"]
ASPECTS = [
    "price",
    "safety",
    "transit-location",
    "general",
]
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

def la2idx(self):
    return {
        (l, a): i * len(LOCATIONS) + j
        for i, l in enumerate(LOCATIONS)
        #for j, a in enumerate(ASPECTS)
        for j, a in enumerate(ALL_ASPECTS)
    }

def unzip(xs):
    return zip(*xs)

# opnions: [{"sentiment": , "aspect": , "location": }]
def get_opinions(opinions):
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
        #for l in LOCATIONS for a in ASPECTS
        for l in LOCATIONS for a in ALL_ASPECTS
    ))

def make_fields():
    SENTIMENT = Field(lower=True, is_target=True)
    TEXT = Field(
        lower=True, include_lengths=True, is_target=True)
        #lower=True, include_lengths=True, init_token="<bos>", eos_token="<eos>", is_target=True)
    return TEXT, SENTIMENT

def build_vocab(f1, f2, d):
    f1.build_vocab(d)
    f2.build_vocab(d)

def nested_items(name, x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from nested_items(f"{name}_{k}", v)
    else:
        yield (name, x)


class SentihoodExample(Example):

    @classmethod
    def fromJson(cls, data, text_field, sentiment_field):
        exs = []
        for x in json.load(data):
            ex = cls()

            locations, aspects, sentiments = get_opinions(x["opinions"])

            setattr(ex, "locations", text_field.preprocess(list(locations)))
            setattr(ex, "aspects", text_field.preprocess(list(aspects)))
            setattr(ex, "sentiments", sentiment_field.preprocess(list(sentiments)))
            setattr(ex, "text", text_field.preprocess(x["text"]))

            exs.append(ex)
        return exs


class SentihoodDataset(Dataset):

    @staticmethod
    def make_fields(text_field, sentiment_field):
        return [
            ("locations", text_field),
            ("aspects", text_field),
            ("sentiments", sentiment_field),
            ("text", text_field),
        ]


    def __init__(
        self, path,
        text_field,
        sentiment_field,
        **kwargs
    ):

        # Sort by length of the text
        self.sort_key = lambda x: len(x.text)

        fields = self.make_fields(text_field, sentiment_field)

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            examples = SentihoodExample.fromJson(f, text_field, sentiment_field)

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
        sentiment_field,
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
            sentiment_field = sentiment_field,
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



if __name__ == "__main__":
    filepath = "/n/rush_lab/jc/code/sentclass/data"
    TEXT, SENTIMENT = make_fields()

    train, valid, test = SentihoodDataset.splits(
        TEXT, SENTIMENT, path=filepath
    )
    TEXT.build_vocab(train)
    SENTIMENT.build_vocab(train)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), batch_size=32, device=torch.device("cuda:0")
    )
    batch = next(iter(train_iter))
    import pdb; pdb.set_trace()
