import torch
import wandb

def train(epoch, criterion, model, optimizer, loader, partition, device = 'cuda'):
    
    total_loss = 0.0

    model.train()

    for batch_idx, (data, target) in enumerate(loader):
        
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() 

    
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
                epoch, total_loss / len(loader.dataset)))
    
    return total_loss / len(loader.dataset)