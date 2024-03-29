import yaml

def parse_config(path):
    class Struct(object):
        def __init__(self, di):
            for a, b in di.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [Struct(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, Struct(b) if isinstance(b, dict) else b)

    with open(path, 'r') as stream:
        try:
            return Struct(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)