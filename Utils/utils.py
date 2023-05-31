import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from Models.model import NeuralNet

# Crear Dataset y DataLoader de PyTorch
class TranslationDataset(Dataset):
    def __init__(self, One_hot_Dataframe, X, indices = []):   #X contains one hot vectors. y contains a dataframe where rows contain the one hot vectors of the translators
      self.Translator2Label  = {translator:idx for idx, translator in enumerate(One_hot_Dataframe["TRANSLATOR"].unique())}
      self.Translator        = One_hot_Dataframe["TRANSLATOR"].map(self.Translator2Label)
      self.Translator_tensor = torch.tensor(self.Translator.values)

      self.indices           = indices
      self.X                 = torch.tensor(X, dtype=torch.float32)           #Tensor containing all features of the a task

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
      idx = self.indices[index]
      return self.X[idx], self.Translator_tensor[idx]
    

def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=True,
                                         pin_memory=True, num_workers=2)
    return loader



def make(One_hot_Dataframe, config, device="cuda"):

    X                  = One_hot_Dataframe.loc[:, One_hot_Dataframe.columns != 'TRANSLATOR'].values # Features
    Translators        = One_hot_Dataframe['TRANSLATOR']                                            # True labels

    train = TranslationDataset(One_hot_Dataframe, X)        # Usage of thise Dataframe for training
    test  = TranslationDataset(One_hot_Dataframe, X)        # Usage of thise Dataframe for testing

    Splits_Train_Test  = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    Train_Test_indices = Splits_Train_Test.split(X, Translators)

    model = NeuralNet(config.input_size, config.num_classes)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss() # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = config.learning_rate, weight_decay = config.weight_decay)
    
    return model, train, test, criterion, optimizer, Train_Test_indices

    
