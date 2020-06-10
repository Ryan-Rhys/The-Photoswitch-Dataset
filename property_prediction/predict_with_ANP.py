# Copyright Ryan-Rhys Griffiths and Aditya Raymond Thawani 2020
# Author: Ryan-Rhys Griffiths and Penelope Jones
"""
Script for training a model to predict properties in the photoswitch dataset using an Attentive Neural Process.
"""

import argparse

from matplotlib import pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from data_utils import transform_data, DataLoader, featurise_mols
from anp_scripts.attentive_np import AttentiveNP


test_set_size = 0.2
y_size = 1

def main(args):
    path_to_save = args.task + '/results/anp/'
    data_loader = DataLoader(args.task, args.path)
    smiles_list, y = data_loader.load_property_data()

    if args.representation == 'fingerprints':
        X = featurise_mols(smiles_list, args.representation)
    elif args.representation == 'fragments':
        X = featurise_mols(smiles_list, args.representation)
    else:
        X = featurise_mols(smiles_list, args.representation)

    # If True we perform Principal Components Regression

    if args.use_pca:
        n_components = 50
    else:
        n_components = None

    num_features = np.shape(X)[1]

    # We define the Gaussian Process Regression Model using the Tanimoto kernel

    m = None

    def objective_closure():
        return -m.log_marginal_likelihood()

    r2_list = []
    rmse_list = []
    mae_list = []

    # We pre-allocate arrays for plotting confidence-error curves

    _, _, _, y_test = train_test_split(X, y, test_size=test_set_size)  # To get test set size
    n_test = len(y_test)

    rmse_confidence_list = np.zeros((args.n_trials, n_test))
    mae_confidence_list = np.zeros((args.n_trials, n_test))

    print('\nBeginning training loop...')
    j = 0  # index for saving results

    for i in range(0, args.n_trials):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=i)

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        #  We standardise the outputs but leave the inputs unchanged

        X_train, y_train, X_test, _, y_scaler = transform_data(X_train, y_train, X_test, y_test,
                                                               n_components=n_components, use_pca=args.use_pca)

        X_train = torch.from_numpy(X_train).float().unsqueeze(dim=0)
        X_test = torch.from_numpy(X_test).float().unsqueeze(dim=0)
        y_train = torch.from_numpy(y_train).float().unsqueeze(dim=0)

        m = AttentiveNP(x_size=X_train.shape[2], y_size=y_size, r_size=args.r_size,
                        det_encoder_hidden_size=args.det_encoder_hidden_size,
                        det_encoder_n_hidden=args.det_encoder_n_hidden,
                        lat_encoder_hidden_size=args.lat_encoder_hidden_size,
                        lat_encoder_n_hidden=args.lat_encoder_n_hidden,
                        decoder_hidden_size=args.decoder_hidden_size,
                        decoder_n_hidden=args.decoder_n_hidden,
                        lr=args.lr, attention_type="multihead")

        print('...training.')

        m.train(X_train, y_train, batch_size=args.batch_size, iterations=args.iterations, print_freq=None)

        # Now, the context set comprises the training x / y values, the target set comprises the test x values.

        y_pred, y_var = m.predict(X_train, y_train, X_test, n_samples=100)

        y_pred = y_scaler.inverse_transform(y_pred)

        # Compute scores for confidence curve plotting.

        ranked_confidence_list = np.argsort(y_var.numpy(), axis=0).flatten()

        for k in range(len(y_test)):
            # Construct the RMSE error for each level of confidence

            conf = ranked_confidence_list[0:k + 1]
            rmse = np.sqrt(mean_squared_error(y_test[conf], y_pred[conf]))
            rmse_confidence_list[i, k] = rmse

            # Construct the MAE error for each level of confidence

            mae = mean_absolute_error(y_test[conf], y_pred[conf])
            mae_confidence_list[i, k] = mae

        # Output Standardised RMSE and RMSE on Train Set

        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        print("\nR^2: {:.3f}".format(score))
        print("RMSE: {:.3f}".format(rmse))
        print("MAE: {:.3f}".format(mae))

        r2_list.append(score)
        rmse_list.append(rmse)
        mae_list.append(mae)

        np.savetxt(path_to_save + '_seed_' + str(j) + '_ypred_' + args.representation + '.txt', y_pred)
        np.savetxt(path_to_save + '_seed_' + str(j) + '_ytest.txt', y_test)
        np.savetxt(path_to_save + '_seed_' + str(j) + '_ystd_' + args.representation + '.txt', np.sqrt(y_var))

        j += 1

    r2_list = np.array(r2_list)
    rmse_list = np.array(rmse_list)
    mae_list = np.array(mae_list)

    print("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))))
    print("mean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))))
    print("mean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list) / np.sqrt(len(mae_list))))

    with open(path_to_save + args.representation + '.txt', 'w+') as f:
        f.write('\n Representation = ' + str(args.representation))
        f.write('\n Task = ' + str(args.task))
        f.write('\n Use PCA? = ' + str(args.use_pca))
        f.write('\n Number of trials = {} \n'.format(args.n_trials))

        f.write('\n Deterministic encoder hidden size = ' + str(args.det_encoder_hidden_size))
        f.write('\n Deterministic encoder number of layers = ' + str(args.det_encoder_n_hidden))
        f.write('\n Latent encoder hidden size = ' + str(args.lat_encoder_hidden_size))
        f.write('\n Latent encoder number of layers = ' + str(args.lat_encoder_n_hidden))
        f.write('\n Decoder hidden size = ' + str(args.decoder_hidden_size))
        f.write('\n Decoder number of layers = ' + str(args.decoder_n_hidden))
        f.write('\n Latent variable size = ' + str(args.r_size))
        f.write('\n Batch size = {}'.format(args.batch_size))
        f.write('\n Learning rate = {}'.format(args.lr))
        f.write('\n Number of iterations = {} \n'.format(args.iterations))


        f.write("\nmean R^2: {:.4f} +- {:.4f}".format(np.mean(r2_list), np.std(r2_list) / np.sqrt(len(r2_list))))
        f.write("\nmean RMSE: {:.4f} +- {:.4f}".format(np.mean(rmse_list), np.std(rmse_list) / np.sqrt(len(rmse_list))))
        f.write("\nmean MAE: {:.4f} +- {:.4f}\n".format(np.mean(mae_list), np.std(mae_list) / np.sqrt(len(mae_list))))

        f.flush()

    # Plot confidence-error curves

    confidence_percentiles = np.arange(1e-14, 100,
                                       100 / len(y_test))  # 1e-14 instead of 0 to stop weirdness with len(y_test) = 29

    rmse_mean = np.mean(rmse_confidence_list, axis=0)
    rmse_std = np.std(rmse_confidence_list, axis=0)

    # We flip because we want the most confident predictions on the right-hand side of the plot

    rmse_mean = np.flip(rmse_mean)
    rmse_std = np.flip(rmse_std)

    # One-sigma error bars

    lower = rmse_mean - rmse_std
    upper = rmse_mean + rmse_std

    plt.plot(confidence_percentiles, rmse_mean, label='mean')
    plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
    plt.xlabel('Confidence Percentile')
    plt.ylabel('RMSE (nm)')
    plt.ylim([0, np.max(upper) + 1])
    plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
    plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
    plt.savefig(path_to_save + 'confidence_curve_rmse.png')

    # We plot the Mean-absolute error confidence-error curves

    mae_mean = np.mean(mae_confidence_list, axis=0)
    mae_std = np.std(mae_confidence_list, axis=0)

    mae_mean = np.flip(mae_mean)
    mae_std = np.flip(mae_std)

    lower = mae_mean - mae_std
    upper = mae_mean + mae_std

    plt.plot(confidence_percentiles, mae_mean, label='mean')
    plt.fill_between(confidence_percentiles, lower, upper, alpha=0.2)
    plt.xlabel('Confidence Percentile')
    plt.ylabel('MAE (nm)')
    plt.ylim([0, np.max(upper) + 1])
    plt.xlim([0, 100 * ((len(y_test) - 1) / len(y_test))])
    plt.yticks(np.arange(0, np.max(upper) + 1, 5.0))
    plt.savefig(path_to_save + 'confidence_curve_mae.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='z_iso_n', help='Task name (thermal, e_iso_pi, '
                                                          'z_iso_pi, e_iso_n, z_iso_n).')
    parser.add_argument('--path', default='../dataset/photoswitches.csv', help='Path to raw dataset.')
    parser.add_argument('--representation', default='fingerprints', help='Descriptor type.')
    parser.add_argument('--use_pca', type=bool, default=True, help='If true, apply PCA to data (50 components).')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of train test splits to try.')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='The number of samples to take of the context set, given the number of'
                             ' context points that should be selected.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The training learning rate.')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of training iterations.')
    parser.add_argument('--r_size', type=int, default=8,
                        help='Dimensionality of context encoding, r.')
    parser.add_argument('--det_encoder_hidden_size', type=int, default=32,
                        help='Dimensionality of deterministic encoder hidden layers.')
    parser.add_argument('--det_encoder_n_hidden', type=int, default=2,
                        help='Number of deterministic encoder hidden layers.')
    parser.add_argument('--lat_encoder_hidden_size', type=int, default=32,
                        help='Dimensionality of latent encoder hidden layers.')
    parser.add_argument('--lat_encoder_n_hidden', type=int, default=2,
                        help='Number of latent encoder hidden layers.')
    parser.add_argument('--decoder_hidden_size', type=int, default=32,
                        help='Dimensionality of decoder hidden layers.')
    parser.add_argument('--decoder_n_hidden', type=int, default=2,
                        help='Number of decoder hidden layers.')
    args = parser.parse_args()

    main(args)