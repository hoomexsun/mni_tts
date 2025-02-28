import os
import yaml


class Dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if isinstance(value, dict):
                value = Dotdict(value)
            self[key] = value


class HParam(Dotdict):
    def __init__(self, file):
        super().__init__()
        hp_dotdict = load_hparam(file)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)

    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__


def load_hparam_str(hp_str: str) -> HParam:
    path = "temp-restore.yaml"
    with open(path, "w") as f:
        f.write(hp_str)
    os.remove(path)
    return HParam(path)


def load_hparam(filename: str) -> Dotdict:
    with open(filename, "r") as stream:
        docs = yaml.safe_load_all(stream)  # Use safe_load_all to retain data types
        hparam_dict = dict()
        for doc in docs:
            if doc is not None:  # Check if doc is not None to avoid issues
                for k, v in doc.items():
                    hparam_dict[k] = v
    return Dotdict(hparam_dict)


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user
