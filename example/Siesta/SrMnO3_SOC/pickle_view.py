import pickle
import sys


def view_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(data.__dict__.keys())
    print(data.wann_names)


view_pickle(sys.argv[1])
