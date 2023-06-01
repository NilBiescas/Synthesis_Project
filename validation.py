import torch
import wandb


@torch.no_grad()  # prevent this function from computing gradients 
def validate(criterion, model, loader, partition, device = 'cuda'):

    val_loss = 0
    correct = 0

    model.eval()

    for batch_idx, (data, target) in enumerate(loader):

        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        val_loss += loss.item()                                                              
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability                                                                 
        correct += pred.eq(target.view_as(pred)).sum().item()

        
        wandb.log({f'Validation loss partition num: {partition}': loss})

    val_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(loader.dataset), accuracy))


    return val_loss, accuracy