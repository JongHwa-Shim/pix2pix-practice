import pickle


def save_dataset(dataset, dataset_path):
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f) #check whether or not epickle.load() load just one piece of dataset
    return dataset

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(model_path):
    with open(model_path):
        model = pickle.load(f)
    return model