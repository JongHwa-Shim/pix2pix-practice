import pickle
import torch

def save_dataset(dataset, dataset_path):
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f) #check whether or not epickle.load() load just one piece of dataset
    return dataset

def save_model_state(model_state, model_path):
    torch.save(model_state, model_path)

def load_model_state(model_class, model_path):
    model = model_class()
    model.load_state_dict(torch.load(model_path))
    return model

def save_model(model, model_path):
    torch.save(model, model_path)

def load_model(model_path):
    model = torch.load(model_path)
    return model