"""
Function for deterministically encoding context points (x, y)_i using a fully connected neural network.
Input = (x, y)_i; output = r_i.
"""

import torch
import torch.nn.functional as F
from torch import nn

from Attentive_NP.attention import dot_product_attention, uniform_attention, laplace_attention, MultiHeadAttention


class DeterministicEncoder(nn.Module):
    """The Deterministic Encoder."""

    def __init__(self, x_size, y_size, r_size, n_hidden,
                 hidden_size, self_att, cross_att,
                 attention_type="uniform"):
        """
        :param input_size: An integer describing the dimensionality of the input to the encoder;
                           in this case the sum of x_size and y_size
        :param r_size: An integer describing the dimensionality of the embedding, r_i
        :param n_hidden: An integer describing the number of hidden layers in the neural
                                 network
        :param hidden_size: An integer describing the number of nodes in each layer of
                                    the neural network
        """
        super().__init__()
        self.input_size = x_size + y_size
        self.x_size = x_size
        self.y_size = y_size
        self.r_size = r_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size
        self.self_att = self_att
        self.cross_att = cross_att
        self.attention_type = attention_type

        self.fcs = nn.ModuleList()
        for i in range(self.n_hidden + 1):
            if i == 0:
                self.fcs.append(nn.Linear(self.input_size, self.hidden_size))

            elif i == self.n_hidden:
                self.fcs.append(nn.Linear(self.hidden_size, self.r_size))

            else:
                self.fcs.append(nn.Linear(self.hidden_size, self.hidden_size))

        if self.self_att:
            print("Deterministic encoder: using multihead self attention.")
            self.self_attention = MultiHeadAttention(key_size=self.hidden_size,
                                                     value_size=self.hidden_size,
                                                     num_heads=4,
                                                     key_hidden_size=self.hidden_size)

        else:
            print("Deterministic encoder: not using self attention.")

        if self.cross_att:
            self.key_transform = nn.ModuleList([nn.Linear(self.x_size, self.hidden_size),
                                                nn.Linear(self.hidden_size, self.hidden_size)])

            if attention_type == "multihead":
                print("Deterministic encoder: using multihead cross attention.")
                self.cross_attention = MultiHeadAttention(key_size=self.hidden_size,
                                                          value_size=self.r_size,
                                                          num_heads=4,
                                                          key_hidden_size=self.r_size,
                                                          normalise=True)
            else:
                print("Deterministic encoder: using uniform cross attention.")

        else:
            print("Deterministic encoder: not using cross attention.")

    def forward(self, x, y, x_target):
        """
        :param x: A tensor of dimensions [batch_size, number of context points
                  N_context, x_size]. In this case each value of x is the concatenation
                  of the input x with the output y
        :param y:
        :param x_target:
        :return: The embeddings, a tensor of dimensionality [batch_size, N_context,
                 r_size]
        """

        input = torch.cat((x, y), dim=-1).float()  # [batch_size, N_context, (x_size + y_size)]

        batch_size = input.shape[0]
        input = input.view(-1, self.input_size)  # [batch_size * N_context, (x_size + y_size)]

        for fc in self.fcs[:-1]:
            input = F.relu(fc(input))  # [batch_size * N_context, hidden_size]

        input = input.view(batch_size, -1, self.hidden_size)  # [batch_size, N_context, hidden_size]
        if self.self_att:
            input = self.self_attention.forward(input)  # [batch_size, N_context, hidden_size]

        input = self.fcs[-1](input)  # [batch_size, N_context, r_size]

        # Aggregate the embeddings
        input = input.view(batch_size, -1, self.r_size)  # [batch_size, N_context, r_size]

        # Using cross attention
        x_target = x_target.view(-1, self.x_size)
        x = x.view(-1, self.x_size)

        # First transform the inputs
        for transform in self.key_transform[:-1]:
            queries = F.relu(transform(x_target))
            keys = F.relu(transform(x))
        queries = self.key_transform[-1](queries)  # [batch_size, N_target, hidden_size]
        keys = self.key_transform[-1](keys)  # [batch_size, N_context, hidden_size]
        queries = queries.view(batch_size, -1, self.hidden_size)
        keys = keys.view(batch_size, -1, self.hidden_size)

        if self.attention_type == "multihead":
            output = self.cross_attention.forward(queries=queries.float(), keys=keys.float(),
                                                  values=input)  # [batch_size, N_target, r_size]

        elif self.attention_type == "uniform":
            output = uniform_attention(queries=queries.float(), values=input)

        elif self.attention_type == "laplace":
            output = laplace_attention(queries=x_target.float(), keys=x.float(), values=input,
                                       scale=1.0, normalise=True)

        elif self.attention_type == "dot_product":
            output = dot_product_attention(queries=x_target.float(), keys=x.float(),
                                           values=input, normalise=True)

        # Otherwise take the mean of the embeddings as for the vanilla NP (same as uniform).
        else:
            output = torch.squeeze(torch.mean(input, dim=1), dim=1)  # [batch_size, self.r_size]
            output = torch.unsqueeze(output, dim=1).repeat(1, x_target.shape[1], 1)
            # [batch_size, # N_target, self.r_size]

        return output
