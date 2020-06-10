"""
Function for encoding context points (x, y)_i using latent space.
Input = (x, y)_i; output = r_i.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from Attentive_NP.attention import MultiHeadAttention


class LatentEncoder(nn.Module):
    """The Latent Encoder."""

    def __init__(self, input_size, r_size, n_hidden, hidden_size, self_att):
        """
        :param input_size: An integer describing the dimensionality of the input to the encoder;
                           in this case the sum of x_size and y_size
        :param r_size: An integer describing the dimensionality of the embedding, r_i
        :param encoder_n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param encoder_hidden_size: An integer describing the number of nodes in each layer of
                                    the neural network
        """

        super().__init__()
        self.input_size = input_size
        self.r_size = r_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.self_att = self_att

        self.n2_hidden = 2
        self.fcs1 = nn.ModuleList()
        self.fcs2 = nn.ModuleList()

        # Encoder function taking as input (x,y)_i and outputting r_i
        for i in range(self.n_hidden + 1):
            if i == 0:
                self.fcs1.append(nn.Linear(input_size, hidden_size))

            elif i == n_hidden:
                self.fcs1.append(nn.Linear(hidden_size, r_size))

            else:
                self.fcs1.append(nn.Linear(hidden_size, hidden_size))

        if self.self_att:
            print("Latent encoder: using multihead self attention.")
            self.self_attention = MultiHeadAttention(key_size=self.hidden_size,
                                                     value_size=self.hidden_size,
                                                     num_heads=4,
                                                     key_hidden_size=self.hidden_size)

        else:
            print("Latent encoder: not using self attention.")

        # For the latent encoder, we also have a second encoder function which takes the
        # aggregated embedding r as an input and outputs a mean and the log(variance)
        for i in range(self.n2_hidden + 1):
            if i == self.n2_hidden:
                self.fcs2.append(nn.Linear(r_size, 2 * r_size))

            else:
                self.fcs2.append(nn.Linear(r_size, r_size))

    def forward(self, x, y):
        """
        :param x: A tensor of dimensions [batch_size, number of context points
                  N_context, x_size + y_size]. In this case each value of x
                  is the concatenation of the input x with the output y
        :return: The embeddings, a tensor of dimensionality [batch_size, N_context,
                 r_size]
        """
        batch_size = x.shape[0]
        input = torch.cat((x, y), dim=-1).float()
        # [batch_size, N_context, (x_size + y_size)]
        # Pass (x, y)_i through the first MLP.
        input = input.view(-1, self.input_size)  # [batch_size * N_context,
        # (x_size + y_size)]
        for fc in self.fcs1[:-1]:
            input = F.relu(fc(input))

        input = input.view(batch_size, -1, self.hidden_size)
        # [batch_size, N_context, hidden_size]
        if self.self_att:
            input = self.self_attention.forward(input)  # [batch_size, N_context, hidden_size]

        input = self.fcs1[-1](input)  # [batch_size, N_context, r_size]

        input = input.view(batch_size, -1, self.r_size)  # [batch_size, N_context, r_size]

        # Aggregate the embeddings
        input = torch.squeeze(torch.mean(input, dim=1), dim=1)  # [batch_size, r_size]

        # Pass the aggregated embedding through the second MLP to obtain means
        # and variances parametrising the distribution over the latent variable z.
        for fc in self.fcs2[:-1]:
            input = F.relu(fc(input))
        input = self.fcs2[-1](input)  # [batch_size, 2*r_size]

        # The outputs are the latent variable mean and log(variance)
        mus_z, ws_z = input[:, :self.r_size], input[:, self.r_size:]
        sigmas_z = 0.001 + 0.999 * F.softplus(ws_z)  # [batch_size, r_size]

        dists_z = [MultivariateNormal(mu, torch.diag_embed(sigma)) for mu, sigma in
                   zip(mus_z, sigmas_z)]

        return dists_z, mus_z, sigmas_z
