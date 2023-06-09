import random
import wandb

import numpy as np
import torch

import pandas as pd

from Utils.organizer import organizer
from Utils.utils_Dataset import process_dataset, OneHotDataframe

from train import *
from test import *                  
from Utils.utils import *     #Contains the functions: (make(), )
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(One_hot_Dataframe, cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="IDISC Lovers", config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # Add the name of the run
        wandb.run.name = config.name

        model = organizer(One_hot_Dataframe, config)

    return model

if __name__ == "__main__":
    wandb.login()

    pkl_file = '/home/xnmaster/Language_GrauIAI_UAB.pkl' #Pickle file path
    # Obtain the dataset
    Task_Data          = pd.read_pickle(pkl_file)               # Read the pkl file containg the pandas dataframe object
    Dataset_process    = process_dataset(Task_Data)             # Obtain the preprocess Dataset
    One_hot_Dataframe  = OneHotDataframe(Dataset_process)       # Changed categorical columns using one hot vectors

    num_classes = len(One_hot_Dataframe["TRANSLATOR"].unique()) # Number of translators
    input_size  = len(One_hot_Dataframe.columns) - 1 

    #Training configurations
    config = dict(
        name="256_Batch_Size_30_epocs",
        epochs=50,
        classes=num_classes,
        batch_size= 256,
        learning_rate= 0.00001,
        weight_decay=0.0001,
        input_size=input_size,
        dim = 256,
        save=True)
    
    model = model_pipeline(One_hot_Dataframe, config)