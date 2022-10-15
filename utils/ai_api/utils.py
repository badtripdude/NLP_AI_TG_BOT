import pathlib


def export_model(ai, path: pathlib.Path):
    return ai.train_model.save_weights(path)


def import_model(ai, path: pathlib.Path = '.'):
    t_m = ai.train_model
    t_m.load_weights(path)
    return AI(t_m, create_module(t_m))
