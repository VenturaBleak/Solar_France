import torch
from tqdm import tqdm
import pickle
import os

def train_fn(loader, model, optimizer, loss_fn, scaler, scheduler, device, epoch, model_path, model_name):
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
    for batch_idx, (X, y) in enumerate(loop):
        # move data to device
        X = X.to(device)

        # move targets to device, convert to float and unsqueeze
        y = y.float().to(device)

        # forward using autocast, which allows for mixed precision, i.e. half precision -> speed up training process
        with torch.cuda.amp.autocast():
            # forward pass
            preds = model(X)
            # calculate loss
            loss = loss_fn(preds, y)

        # zero out gradients
        optimizer.zero_grad()

        # scaler.scale(loss).backward() is used to scale the loss and backpropagate
        #scaler.scale(loss).backward()
        # loss.backward() is used to backpropagate without scaling the loss
        loss.backward()

        # Clip the gradients to prevent them from exploding
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # scaler.step(optimizer) is used to scale the gradients
        # scaler.step(optimizer)
        # optimizer.step() is used to backpropagate without scaling the gradients
        optimizer.step()

        # scaler.update() is used to update the scale for next iteration
        # scaler.update()


        epoch_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=f"{loss.item():.4f}")

        # Save the gradients at the first batch of every 20th epoch
        if epoch % 20 == 0 and batch_idx == 0:
            gradients_dict = {}  # Instantiate the dict here
            for name, param in model.named_parameters():
                if param.requires_grad:
                    gradients_dict[name] = param.grad.clone().detach().cpu().numpy() # Store the gradients in the dict
            with open(os.path.join(model_path,f'{model_name}_Epoch{epoch}_gradients.pkl'), 'wb') as handle:  # Save dict to file
                pickle.dump(gradients_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # step scheduer on batch
        if scheduler.is_batch == True:
            scheduler.step()

    # step scheduler on epoch
    if scheduler.is_batch == False:
        scheduler.step()

    # calculate average epoch loss
    epoch_loss = epoch_loss / num_batches

    # update tqdm loop
    loop.set_postfix(loss=f"{epoch_loss:.4f}")

    return epoch_loss