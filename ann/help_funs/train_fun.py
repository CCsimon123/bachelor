import time
import torch


def train_nn(number_of_epochs, net, train_dataloader, optimizer, loss_func, val_dataloader):
    start_time = time.time()
    # train the network
    for epoch in range(number_of_epochs):
        torch.manual_seed(1 + epoch)  # recovery reproducibility
        epoch_loss_train = 0.0
        epoch_loss_val = 0.0

        net.train()
        for (idx, X) in enumerate(train_dataloader):
            (input_data, label) = X
            optimizer.zero_grad()
            output = net(input_data)
            output = torch.squeeze(output)
            train_loss = loss_func(output, label)
            epoch_loss_train += train_loss.item()
            train_loss.backward()
            optimizer.step()

        net.eval()
        for (idx, X) in enumerate(val_dataloader):
            with torch.no_grad():
                (input_data, label) = X
                output = net(input_data)
                output = torch.squeeze(output)
                val_loss = loss_func(output, label)
                epoch_loss_val += val_loss.item()

        if epoch % 1 == 0 or epoch == number_of_epochs - 1:
            print(f'epoch {epoch} train loss = {epoch_loss_train}')
            print(f'epoch {epoch}   val loss = {epoch_loss_train}')

    end_time = time.time()
    print(f'time for the training: {end_time - start_time}')