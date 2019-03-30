__author__ = 'YuxuanXiu'
import torch
from torch import optim
from torch.autograd import Variable
from experimental_design import train_batch_size, learning_rate, num_epoches, patience
from physical_guided_neural_network import PhysicalGuidedNeuralNetwork
from loss_function import empirical_error,structural_error,physical_inconsistency_batch
from data_preparation import train_data_loader, val_data_loader, test_data_loader
import numpy as np
from pytorchtools import EarlyStopping


if __name__ == "__main__":

    model = PhysicalGuidedNeuralNetwork(12, 12, 12, 12, 1)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_data_loader, 1):
            input_variables_data_loader, label_data_loader = data
            if torch.cuda.is_available():
                input_variables = Variable(input_variables_data_loader).cuda()
                label = Variable(label_data_loader).cuda()
            else:
                input_variables = Variable(input_variables_data_loader)
                label = Variable(label_data_loader)
            # Calculate forward
            out = model(input_variables)

            # Prepare for PHY_loss

            Y_ = out.to(torch.device("cpu"))
            Y_ = Y_.numpy()
            input_variables_np = input_variables_data_loader.numpy()
            label_np = label_data_loader.numpy()
            batch_data = np.c_[input_variables_np, label_np, Y_]
            PHY_loss = physical_inconsistency_batch(batch_data)
            PHY_loss = torch.from_numpy(PHY_loss)

            if torch.cuda.is_available():
                PHY_loss = Variable(PHY_loss).cuda()
            else:
                PHY_loss = Variable(PHY_loss)

            loss = empirical_error(out, label) + structural_error(model, 1, 1) + PHY_loss

            if torch.cuda.is_available():
                loss_cpu = loss.data.cpu()
            else:
                loss_cpu = loss.data
            running_loss += loss_cpu.numpy() * label.size(0)

            # gradient backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data, target in val_data_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            input_variables_data_loader, label_data_loader = data
            if torch.cuda.is_available():
                input_variables = Variable(input_variables_data_loader).cuda()
                label = Variable(label_data_loader).cuda()
            else:
                input_variables = Variable(input_variables_data_loader)
                label = Variable(label_data_loader)
            # Calculate forward
            out = model(input_variables)
            # calculate the loss
            # Prepare for PHY_loss

            Y_ = out.to(torch.device("cpu"))
            Y_ = Y_.numpy()
            input_variables_np = input_variables_data_loader.numpy()
            label_np = label_data_loader.numpy()
            batch_data = np.c_[input_variables_np, label_np, Y_]
            PHY_loss = physical_inconsistency_batch(batch_data)
            PHY_loss = torch.from_numpy(PHY_loss)

            if torch.cuda.is_available():
                PHY_loss = Variable(PHY_loss).cuda()
            else:
                PHY_loss = Variable(PHY_loss)

            loss = empirical_error(out, label) + structural_error(model, 1, 1) + PHY_loss

            if torch.cuda.is_available():
                loss_cpu = loss.data.cpu()
            else:
                loss_cpu = loss.data
            # record validation loss
            valid_losses.append(loss_cpu)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(num_epoches))

        print_msg = '[{}:>{}/{}:>{}] '.format(epoch,epoch_len,num_epochs,epoch_len) +\
        'train_loss: %.5f '% train_loss +\
        'valid_loss: %.5f'% valid_loss

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        if i % 300 == 0:
            print('[{}/{}] Loss: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (train_batch_size * i)))
        print('Finish {} epoch'.format(epoch + 1))

            # load the last checkpoint with the best model
    torch.save(model.state_dict(), './neural_network.pth')
    model.load_state_dict(torch.load('checkpoint.pt'))



