from Utils.utils import make, make_loader
import wandb

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
        # Explain this
        train_data.indices = train_indices      
        test_data.indices  = test_indices

        val_loss_partition = 100000
        for epoch in range(config.epochs):
            train_loss = train(epoch, criterion, model, optimizer, train_loader)
            val_loss   = validate(criterion, model, val_loader)

            # Log the train and validation loss values to wandb

            if val_loss < val_loss_partition:
                val_loss_partition = val_loss
            wandb.log({'Loss ' + str(partition): train_loss, 'Split': 'Train'})
            wandb.log({'Loss ' + str(partition): val_loss,   'Split': 'Validation'})

        model = reset_weights(model)
        #Reset the model at each iteration








    # and use them to train the model