"""
Module containing attention definitions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, n_hidden, output_size, num_heads = 4):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.num_heads = num_heads

        self.initial_transform = nn.Linear(self.input_size, self.hidden_size)
        self.pre_relu_transform = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.post_relu_transform = nn.ModuleList()
        self.final_transform = nn.Linear(self.hidden_size, self.output_size)

        for i in range(self.n_hidden - 1):
            self.attentions.append(MultiHeadAttention(key_size=self.hidden_size, value_size=self.hidden_size,
                                                      key_hidden_size=self.hidden_size, num_heads=self.num_heads))
            self.pre_relu_transform.append(nn.Linear(self.hidden_size, self.hidden_size))

            self.post_relu_transform.append(nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self, input, batch_size):  # input = [batch_size * N_context, input_size]
        self.batch_size = batch_size
        # Initial transformation
        input = F.relu(self.initial_transform(input)).view(self.batch_size, -1, self.hidden_size)  # [batch_size, N_context, hidden_size].

        # There are sublayers within each layer; the first is a Multihead attention sublayer, the second is a FFN
        # output = Relu(W1*input + b1)*W2 + b2 (as described in "Attention Is All You Need" paper).
        # Need to add the Layer Norm and mask.
        for attention, fc1, fc2 in zip(self.attentions, self.layer_norms, self.pre_relu_transform, self.post_relu_transform):
            input = attention(input)

            input = F.relu(fc1(input)) # [batch_size, N_context, hidden_size].

            input = fc2(input)

        # input = input.view(-1, self.hidden_size)
        input = self.final_transform(input)

        return input


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention class
    """
    def __init__(self, key_size, value_size, num_heads, key_hidden_size, normalise=True):
        """
        :param num_heads:
        :param normalise:
        """
        super().__init__()
        self._key_size = key_size
        self._value_size = value_size
        self._num_heads = num_heads
        self._key_hidden_size = key_hidden_size
        self._head_size = int(self._value_size / self._num_heads)
        self._normalise = normalise

        self._query_transform = nn.Linear(self._key_size, self._num_heads*self._key_hidden_size, bias=False)  # Apply linear transformation [key_size, key_hidden_size]
        self._key_transform = nn.Linear(self._key_size, self._num_heads*self._key_hidden_size, bias=False)  # Apply linear transformation [key_size, key_hidden_size]
        self._value_transform = nn.Linear(self._value_size, self._num_heads*self._head_size, bias=False)  # Apply linear transformation [value_size, head_size]
        self._head_transform = nn.Linear(self._num_heads*self._head_size, self._value_size, bias=False)  # Apply final linear transformation [num_heads*head_size, value_size]

    def forward(self, queries, keys=None, values=None):
        """
        :param queries: [batch_size, N_target, key_size]
        :param keys: [batch_size, N_context, key_size]
        :param values: [batch_size, N_context, value_size]
        :return:
        """

        # For self-attention mechanism, queries, keys and values all take the same values.
        if keys is None:
            keys = queries

        if values is None:
            values = queries

        self._batch_size = queries.shape[0]
        self._n_target = queries.shape[1]
        self._n_context = keys.shape[1]

        # Linearly transform the queries, keys and values:
        queries = self._query_transform(queries).view(self._batch_size, self._n_target, self._num_heads, self._key_hidden_size)
        keys = self._key_transform(keys).view(self._batch_size, self._n_context, self._num_heads, self._key_hidden_size)
        values = self._value_transform(values).view(self._batch_size, self._n_context, self._num_heads, self._head_size)

        # Transpose so that in form [batch_size, num_heads, ...]
        queries = queries.transpose(1, 2)  # [batch_size, num_heads, N_target, key_hidden_size]
        keys = keys.transpose(1, 2)  # [batch_size, num_heads, N_context, key_hidden_size]
        values = values.transpose(1, 2)  # [batch_size, num_heads, N_context, head_size]

        attention = dot_product_attention(queries, keys, values, normalise=self._normalise)  # [batch_size, num_heads, N_target, head_size]

        attention = attention.transpose(1, 2)  # [batch_size, N_target, num_heads, head_size]
        attention = attention.reshape(self._batch_size, self._n_target, -1)  # [batch_size, N_target, num_heads*head_size]
        output = self._head_transform(attention)  # [batch_size, N_target, value_size]

        return output


def uniform_attention(queries, values):
    """
    In the case of uniform attention, the weight assigned to each value is independent of the value of the
    corresponding key; we can simply take the average of all of the values. This is the equivalent of the "vanilla"
    neural process, where r* is the average of the context set embeddings.

    :param queries: Queries correspond to x_target. [batch_size, N_target, key_size]
    :param values: Values corresponding to the aggregated embeddings r_i. [batch_size, N_context, value_size]
    :return:
    """

    N_target = queries.shape[1]
    attention = torch.mean(values, dim=1, keepdim=True) # [batch_size, 1, value_size]
    output = attention.repeat(1, N_target, 1)  # [batch_size, N_target, value_size]

    return output


def laplace_attention(queries, keys, values, scale, normalise=True):
    """
    Here we compute the Laplace exponential attention. Each value is weighted by an amount that depends
    on the distance of the query from the corresponding key (specifically, w_i ~ exp(-||q-k_i||/scale))
    :param queries: e.g. query corresponding to x_target: [batch_size, N_target, key_size]
    :param keys: e.g. x_context: [batch_size, N_context, key_size]
    :param values: e.g. values corresponding to the aggregated embeddings r_i [batch_size, N_context, value_size]
    :param scale: float value which scales the L1 distance.
    :param normalise: Boolean, determines whether we should normalise s.t. sum of weights = 1.
    :return: A tensor [batch_size, N_target, value_size].
    """

    keys = torch.unsqueeze(keys, dim=1)  # [batch_size, 1, N_context, key_size]
    queries = torch.unsqueeze(queries, dim=1)  # [batch_size, N_target, 1, key_size]

    unnorm_weights = -torch.abs((keys - queries)/scale)  # [batch_size, N_target, N_context, key_size]
    unnorm_weights = torch.sum(unnorm_weights, dim=-1, keepdim=False) # [batch_size, N_target, N_context]

    if normalise:
        attention = torch.softmax(unnorm_weights, dim=-1)  # [batch_size, N_target, N_context]
    else:
        attention = 1 + torch.tanh(unnorm_weights)  # [batch_size, N_target, N_context]

    # Einstein summation over weights and values
    output= torch.matmul(attention, values)  # [batch_size, N_target, value_size]

    return output


def dot_product_attention(queries, keys, values, normalise=True):
    """
    :param queries:[batch_size, N_target, key_size]
    :param keys:[batch_size, N_context, key_size]
    :param values: []
    :param normalise:
    :return:
    """
    key_size = keys.shape[-1]
    scale = np.sqrt(key_size)

    unnorm_weights = torch.matmul(queries, keys.transpose(-2, -1)) / scale  # [batch_size, N_target, N_context]

    if normalise:
        attention = torch.softmax(unnorm_weights, dim=-1)

    else:
        attention = torch.sigmoid(unnorm_weights)  # [batch_size, N_target, N_context]

    # Einstein summation over weights and values
    output = torch.matmul(attention, values)  # [batch_size, N_target, value_size]
    return output
