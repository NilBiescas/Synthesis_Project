from Utils.utils import make, make_loader
import wandb
from tqdm.auto import tqdm

import torch.nn as nn
from train       import *
from validation  import validate

def reset_weights(model):
    # Reset the parameters (weights) of each module in the model
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.reset_parameters()
    return model

def organizer(One_hot_Dataframe, config):
    # make the model, data, and optimization problem
    model, train_data, test_data, criterion, optimizer, Train_Test_indices = make(One_hot_Dataframe, config, device = 'cuda')

    train_loader  = make_loader(train_data, batch_size=config.batch_size)
    val_loader    = make_loader(test_data,  batch_size=config.batch_size)

    total_val_loss = 100000

    for partition, (train_indices, test_indices) in enumerate(Train_Test_indices):
        print('Starting partition number: ',partition)
        # Explain this
        train_data.indices = train_indices      
        test_data.indices  = test_indices

        val_loss_partition = 100000
        
        for epoch in tqdm(range(config.epochs)):
            train_loss           = train(epoch, criterion, model, optimizer, train_loader, partition, device = 'cuda')
            val_loss, accuracy   = validate(criterion, model, val_loader, partition, device = 'cuda')

            wandb.log({f'Validation Loss num: {partition}' : val_loss})
            wandb.log({f'Train Loss num: {partition}' : train_loss})
            wandb.log({f'Accuracy partition num: {partition}' : accuracy}) #At each epoch stored the accuracy 
            
            if val_loss < val_loss_partition:
                val_loss_partition = val_loss

        print(f"Partition number {partition} has finished")

        if val_loss_partition < total_val_loss:
            print("Best model actualized") 
            total_val_loss = val_loss_partition
            best_model = model

    if config.save:
        torch.save(model.state_dict(), f'/home/xnmaster/Synthesis_Project/CheckPoints/{config.name}.pth')

    return best_model