
import torch

import torchtext
from torchtext import data
from torchtext.data import Dataset, Field, Example, Iterator, TabularDataset

import io
import os
import json

def make_fields():
    SENTIMENT = Field()
    TEXT = Field(
        lower=True, include_lengths=True, is_target=True)
        #lower=True, include_lengths=True, init_token="<bos>", eos_token="<eos>", is_target=True)
    return TEXT, SENTIMENT

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

            locations  = []
            aspects    = []
            sentiments = []

            for op in x["opinions"]:
                location  = op["target_entity"]
                aspect    = op["aspect"]
                sentiment = op["sentiment"]
                locations.append(location)
                aspects.append(aspect)
                sentiments.append(sentiment)

            setattr(ex, "locations", text_field.preprocess(locations))
            setattr(ex, "aspects", text_field.preprocess(aspects))
            setattr(ex, "sentiments", sentiment_field.preprocess(sentiments))
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
