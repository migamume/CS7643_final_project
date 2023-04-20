import pickle

from holdup.the_model.get_datasets import get_datasets


def load_and_pickle():
    data = get_datasets(last_possible=True)
    with open('last_possible.pickle', 'wb') as file:
        pickle.dump(data, file)

    data = get_datasets(last_possible=False)
    with open('last_action.pickle', 'wb') as file:
        pickle.dump(data, file)


if __name__ == "__main__":
    load_and_pickle()