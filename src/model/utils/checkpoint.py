import torch


def save_checkpoint(name, epoch, model, optimizer, valid_loss, train_loss):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'valid_loss': valid_loss,
            'train_loss': train_loss,
            }, name)

def load_checkpoint(filename):
    checkpoint = torch.load(filename)
    return checkpoint['epoch'], checkpoint['model_state_dict'],\
           checkpoint['optimizer_state_dict'], checkpoint['valid_loss'], checkpoint['train_loss']