import torch
from tqdm import tqdm

def train_fn(loader, model, optimizer, loss_fn, scaler, scheduler, device, epoch):
    """Train function for training the model

    :param loader   (torch.utils.data.DataLoader): training dataloader
    :param model    (torch.nn.Module): model to train
    :param optimizer(torch.optim): optimizer to use

    :param loss_fn  (torch.nn.Module): loss function to use
    :param scaler   (torch.cuda.amp.GradScaler): scaler to use for mixed precision training
    :param device   (torch.device): device to train on (CPU or GPU)
    :param epoch    (int): current epoch number

    :return: None
    """

    # set tqdm loop with epoch number
    loop = tqdm(loader, desc=f"Epoch {epoch}")

    # set model to train mode
    model.train()

    epoch_loss = 0
    num_batches = len(loader)

    # iterate over batches
    for batch_idx, (data, targets) in enumerate(loop):
        # move data to device
        data = data.to(device)

        # move targets to device, convert to float and unsqueeze
        targets = targets.float().unsqueeze(1).to(device)

        # forward using autocast, which allows for mixed precision, i.e. half precision -> speed up training process
        with torch.cuda.amp.autocast():
            # forward pass
            predictions = model(data)
            # calculate loss
            loss = loss_fn(predictions, targets.squeeze(1))

        # zero out gradients
        optimizer.zero_grad()

        # scaler.scale(loss).backward() is used to scale the loss and backpropagate
        scaler.scale(loss).backward()

        # scaler.step(optimizer) is used to scale the gradients
        scaler.step(optimizer)

        # scaler.update() is used to update the scale for next iteration
        scaler.update()

        epoch_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=f"{loss.item():.4f}")

        # update learning rate
        scheduler.step()

    # calculate average epoch loss
    epoch_loss = epoch_loss / num_batches

    # update tqdm loop
    loop.set_postfix(loss=f"{epoch_loss:.4f}")