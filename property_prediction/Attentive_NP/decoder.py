"""
Function for 'decoding' an input vector x_i (conditioned on a set of context points) to obtain a
distribution over y_i. The conditioning is achieved by concatenating x_i with
the deterministic and latent embeddings, r and z. The function comprises a fully connected neural network,
with size and number of hidden layers being hyperparameters to be selected.

Input = (x_i, r, z). Output = (mean_i, var_i)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class Decoder(nn.Module):
    """The Decoder."""
    def __init__(self, input_size, output_size, decoder_n_hidden, decoder_hidden_size):
        """
        :param input_size: An integer describing the dimensionality of the input, in this case
                           (r_size + x_size), where x_size is the dimensionality of x and r_size
                           is the dimensionality of the embedding r
        :param output_size: An integer describing the dimensionality of the output, in this case
                            y_size, which is the dimensionality of the target variable y
        :param decoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param decoder_hidden_size: An integer describing the number of nodes in each layer of the
                                    neural network
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = decoder_n_hidden
        self.hidden_size = decoder_hidden_size

        self.fcs = nn.ModuleList()

        for i in range(decoder_n_hidden + 1):
            if i == 0:
                self.fcs.append(nn.Linear(input_size, decoder_hidden_size))

            elif i == decoder_n_hidden:
                self.fcs.append(nn.Linear(decoder_hidden_size, 2*output_size))

            else:
                self.fcs.append(nn.Linear(decoder_hidden_size, decoder_hidden_size))

    def forward(self, x, r, z):
        """

        :param x: A tensor of dimensions [batch_size, N_target, x_size], containing the input
                  values x_target.
        :param r: A tensor of dimensions [batch_size, N_target, r_size]
                                     containing the deterministic embedding r
        :param z: A tensor of dimensions [batch_size, z_size]
                                     containing the latent embedding z
        :return: A tensor of dimensionality [batch_size, N_target, output_size]
                    describing the distribution over target values y.
        """

        batch_size = x.shape[0]

        # Concatenate the input vectors x_i to the aggregated embedding r.
        z = torch.unsqueeze(z, dim=1).repeat(1, x.shape[1], 1)

        # The input to the decoder is the concatenation of each x_target value with the deterministic
        # embedding r, and the latent embedding z.
        x = torch.cat((x, r), dim=2)   # [batch_size, N_target, (x_size + r_size)]
        x = torch.cat((x, z), dim=2)   # [batch_size, N_target, (x_size + r_size + z_size)]
        x = x.view(-1, self.input_size)   # [batch_size * N_target, (x_size + r_size + z_size)]

        # Pass input through the MLP. The output is the predicted values of y for each value of x.
        for fc in self.fcs[:-1]:
            x = F.relu(fc(x))
        x = self.fcs[-1](x)     # x = [batch_size * N_target, 2*output_size]

        # The outputs are the predicted y means and variances
        mus, log_sigmas = x[:, :self.output_size], x[:, self.output_size:]
        sigmas = 0.001 + 0.999 * F.softplus(log_sigmas) # mu, sigma = [batch_size * N_target, output_size]

        mus = mus.view(batch_size, -1, self.output_size)
        # [batch_size, N_target, output_size]
        sigmas = sigmas.view(batch_size, -1, self.output_size)
        # [batch_size, N_target, output_size]
        dists = [MultivariateNormal(mu, torch.diag_embed(sigma)) for mu, sigma in
                 zip(mus, sigmas)]

        return dists, mus, sigmas
