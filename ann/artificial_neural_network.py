import torch
from torch.utils.data import DataLoader
from numpy.random import RandomState
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from ann.help_funs.plot_fun import *
from ann.help_funs.data_funs import *
from ann.help_funs.train_fun import *
from ann.help_funs.evaluate_funs import *

# Created by Simon Carlson April 2022


def main():
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ setup data ------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # Flags
    create_new_data = False
    load_model = True
    save_model = True
    plot_data = False

    # Some hyper parameters
    train_data_frac = 0.9
    batch_size = 5000
    learning_rate = 0.00005
    number_of_epochs = 2
    num_of_input = 3

    # Paths
    path_to_save_model_to = r'saved_models\final_model_azm_no_feature3.pth'
    path_to_load_from = r'saved_models\final_model_azm_no_feature3.pth'
    path_to_data = r'data\params_inc_azi.csv'
    train_path = Path(r'data\train_data.csv')
    val_path = Path(r'data\val_data.csv')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print('Running on the GPU')
    else:
        device = torch.device("cpu")
        print('Running on the CPU')

    if create_new_data:
        create_data(path_to_data, train_data_frac, train_path, val_path)

    # to get results that can be repeated
    torch.manual_seed(4)
    np.random.seed(4)

    # loads the data from csv-files
    training_data = np.loadtxt(train_path, dtype=np.float32, delimiter=",", skiprows=1)
    training_data = torch.from_numpy(training_data)
    train_data = CustomCsvDataset(training_data, device)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    validation_data = np.loadtxt(val_path, dtype=np.float32, delimiter=",", skiprows=1)
    validation_data = torch.from_numpy(validation_data)
    val_data = CustomCsvDataset(validation_data, device)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    net = Net(num_of_input).to(device)
    # loads the old model
    if load_model:
        net.load_state_dict(torch.load(path_to_load_from, map_location=device))

    # define loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Train the model -------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    train_nn(number_of_epochs, net, train_dataloader, optimizer, loss_func, val_dataloader)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------ Evaluate the model ----------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # Evaluate the trained model
    net.eval()
    train_acc = accuracy(net, train_data, ok_error=0.2)
    print(f'train accuracy: {train_acc}')
    val_acc = accuracy(net, val_data, ok_error)
    print(f'validation ({ok_error}m) accuracy: {val_acc}')

    # save the model
    if save_model:
        torch.save(net.state_dict(), path_to_save_model_to)

    target_arr, model_guess_arr = get_model_guess_vs_target_arr(net, val_data)
    # Calculates RMSE, R-value and MAE
    mse = mean_squared_error(target_arr, model_guess_arr)
    rmse = np.sqrt(mse)
    print(f"RMSE for full validation set: {rmse:.3f}")

    res = r2_score(target_arr, model_guess_arr)
    res = np.sqrt(res)
    print(f"R-value for full validation set: {res:.3f}")

    mean_absolute_error = np.mean(np.abs(target_arr - model_guess_arr))
    print(f"Mean absolute error: {mean_absolute_error:.3f}")

    bias_error = np.mean(target_arr - model_guess_arr)
    print(f"bias error: {bias_error:.3f}")

    if plot_data:
        plot_hist2d(target_arr, model_guess_arr)

if __name__ == '__main__':
    main()


