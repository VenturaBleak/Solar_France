import torch
from tqdm import tqdm


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    """Train function for training the model

    :param loader   (torch.utils.data.DataLoader): training dataloader
    :param model    (torch.nn.Module): model to train
    :param optimizer(torch.optim): optimizer to use

    :param loss_fn  (torch.nn.Module): loss function to use
    :param scaler   (torch.cuda.amp.GradScaler): scaler to use for mixed precision training
        - The scaler is an instance of torch.cuda.amp.GradScaler.
        - The scaler's purpose is to perform gradient scaling when training the model using mixed precision.
        - Mixed precision training allows you to use a combination of lower (half) and higher (single) precision
            floating-point formats for computations, which can result in improved training speed, lower memory usage,
            and similar model performance compared to full-precision training.

    :return: None
    """

    # set tqdm loop
    loop = tqdm(loader)

    # set model to train mode
    model.train()

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
            loss = loss_fn(predictions, targets)

        # zero out gradients
        optimizer.zero_grad()

        # scaler.scale(loss).backward() is used to scale the loss and backpropagate
        scaler.scale(loss).backward()

        # scaler.step(optimizer) is used to scale the gradients
        scaler.step(optimizer)

        # scaler.update() is used to update the scale for next iteration
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())