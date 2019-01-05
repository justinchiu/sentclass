
import torch

import torchtext
from torchtext import data
from torchtext.data import Dataset, Field, Example, Iterator, TabularDataset

import io
import os
import json

def make_fields():
    ENT = Field(lower=True, include_lengths=True)
    TYPE = Field(lower=True, include_lengths=True)
    VALUE = Field(lower=True, include_lengths=True)
    TEXT = Field(
        lower=True, include_lengths=True, init_token="<bos>", eos_token="<eos>", is_target=True)
    return ENT, TYPE, VALUE, TEXT

def build_vocab(a,b,c,d, data):
    a.build_vocab(data)
    b.build_vocab(data)
    c.build_vocab(data)
    d.build_vocab(data)


def nested_items(name, x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from nested_items(f"{name}_{k}", v)
    else:
        yield (name, x)


class RotoExample(Example):
    @classmethod
    def fromJson(cls, data, ent_field, type_field, value_field, text_field):
        exs = []
        for x in json.load(data):
            ex = cls()

            entities = []
            types = []
            values = []

            # Need to flatten all the tables into aligned lists of
            # entities, types, and values.
            # team stuff
            home_name = x["home_name"]
            vis_name = x["vis_name"]

            def add(entity, type, value):
                entities.append(entity)
                types.append(type)
                values.append(value)

            # flat team stats
            add(home_name, "home_city", x["home_city"])
            add(vis_name, "vis_city", x["vis_city"])
            add("day", "day", x["day"])

            # team lines
            for k, v in x["home_line"].items():
                add(home_name, k, v)
            for k, v in x["vis_line"].items():
                add(vis_name, k, v)

            # flatten box_score: {key: {ID: value}}
            box_score = x["box_score"]
            id2name = box_score["PLAYER_NAME"]

            for k, d in box_score.items():
                for id, v in d.items():
                    add(id2name[id], k, v)

            # entities, types, values, summary
            setattr(ex, "entities", ent_field.preprocess(entities))
            setattr(ex, "types", type_field.preprocess(types))
            setattr(ex, "values", value_field.preprocess(values))
            setattr(ex, "text", text_field.preprocess(x["summary"]))

            exs.append(ex)
        return exs


class RotoDataset(Dataset):

    @staticmethod
    def make_fields(entity_field, type_field, value_field, text_field):
        return [
            ("entities", entity_field),
            ("types", type_field),
            ("values", value_field),
            ("text", text_field),
        ]


    def __init__(
        self, path,
        entity_field, type_field, value_field, text_field,
        **kwargs
    ):

        # Sort by length of the text
        self.sort_key = lambda x: len(x.text)

        fields = self.make_fields(entity_field, type_field, value_field, text_field)

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            examples = RotoExample.fromJson(f, entity_field, type_field, value_field, text_field)

        # unused
        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(RotoDataset, self).__init__(examples, fields, **kwargs)


    @classmethod
    def splits(
        cls,
        entity_field, type_field, value_field, text_field,
        path = None,
        root='.data',
        train='train.json', validation='valid.json', test='test.json',
        **kwargs
    ):
        return super(RotoDataset, cls).splits(
            path = path,
            root = root,
            train = train,
            validation = validation,
            test = test,
            entity_field = entity_field,
            type_field = type_field,
            value_field = value_field,
            text_field = text_field,
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
    filepath = "/n/rush_lab/jc/code/data2text/boxscore-data/rotowire/"
    ENT = Field(lower=True, include_lengths=True)
    TYPE = Field(lower=True, include_lengths=True)
    VALUE = Field(lower=True, include_lengths=True)
    TEXT = Field(lower=True, include_lengths=True)
    ENT, TYPE, VALUE, TEXT = make_fields()

    train, valid, test = RotoDataset.splits(
        ENT, TYPE, VALUE, TEXT, path=filepath
    )
    ENT.build_vocab(train)
    TYPE.build_vocab(train)
    VALUE.build_vocab(train)
    TEXT.build_vocab(train)

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), batch_size=32, device=torch.device("cuda:0")
    )
    batch = next(iter(train_iter))
    import pdb; pdb.set_trace()
