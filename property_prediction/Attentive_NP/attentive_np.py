"""
The Attentive Neural Process model.
"""
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.kl import kl_divergence

from Attentive_NP.deterministic_encoder import DeterministicEncoder
from Attentive_NP.latent_encoder import LatentEncoder
from Attentive_NP.decoder import Decoder


class AttentiveNP():
    """
    The Attentive Neural Process model.
    """
    def __init__(self, x_size, y_size, r_size, det_encoder_hidden_size, det_encoder_n_hidden,
                 lat_encoder_hidden_size, lat_encoder_n_hidden, decoder_hidden_size,
                 decoder_n_hidden, lr, attention_type):
        """
        :param x_size: An integer describing the dimensionality of the input x
        :param y_size: An integer describing the dimensionality of the target variable y
        :param r_size: An integer describing the dimensionality of the embedding / context
                       vector r
        :param det_encoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the deterministic encoder NN
        :param det_encoder_n_hidden: An integer describing the number of hidden layers in the
                                 deterministic encoder neural network
        :param lat_encoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the latent encoder neural NN
        :param lat_encoder_n_hidden: An integer describing the number of hidden layers in the
                                 latent encoder neural network
        :param decoder_hidden_size: An integer describing the number of nodes per hidden
                                    layer in the decoder neural network
        :param decoder_n_hidden: An integer describing the number of hidden layers in the
                                 decoder neural network
        :param lr: The optimiser learning rate.
        :param attention_type: The type of attention to be used. A string, either "multihead",
                                "laplace", "uniform", "dot_product"
        """

        self.r_size = r_size
        self.det_encoder = DeterministicEncoder(x_size, y_size, r_size, det_encoder_n_hidden,
                                                det_encoder_hidden_size, self_att=True,
                                                cross_att=True, attention_type=attention_type)

        self.lat_encoder = LatentEncoder((x_size + y_size), r_size, lat_encoder_n_hidden,
                                         lat_encoder_hidden_size, self_att=True)
        self.decoder = Decoder((x_size + r_size + r_size), y_size, decoder_n_hidden,
                               decoder_hidden_size)
        self.optimiser = optim.Adam(list(self.det_encoder.parameters()) +
                                    list(self.lat_encoder.parameters()) +
                                    list(self.decoder.parameters()), lr=lr)

    def train(self, x_trains, y_trains, batch_size=1, iterations=1000, print_freq=None):
        """
        :param x_trains: A np.array with dimensions [N_functions, [N_train, x_size]]
                         containing the training data (x values)
        :param y_trains: A np.array with dimensions [N_functions, [N_train, y_size]]
                         containing the training data (y values)
        :param x_tests: A tensor with dimensions [N_functions, [N_test, x_size]]
                        containing the test data (x values)
        :param y_tests: A tensor with dimensions [N_functions, [N_test, y_size]]
                        containing the test data (y values)
        :param x_scalers: The standard scaler used when testing == True to convert the
                         x values back to the correct scale.
        :param y_scalers: The standard scaler used when testing == True to convert the predicted
                         y values back to the correct scale.
        :param batch_size: An integer describing the number of times we should
                           sample the set of context points used to form the
                           aggregated embedding during training, given the number
                           of context points to be sampled N_context. When testing
                           this is set to 1
        :param iterations: An integer, describing the number of iterations. In this case it
                           also corresponds to the number of times we sample the number of
                           context points N_context
        :param testing: A Boolean object; if set to be True, then every 30 iterations the
                        R^2 score and RMSE values will be calculated and printed for
                        both the train and test data
        :param print_freq:
        :param dataname:
        :param plotting:
        :return:
        """

        n_functions = len(x_trains)  # Should be 1 in this case, as we are effectively bootstrapping to train.

        # If only one function, no need to sample the function index and values every time.
        if n_functions == 1:
            idx_function = 0
            x_train = x_trains[idx_function]
            y_train = y_trains[idx_function]

            max_target = x_train.shape[0]

        for iteration in range(iterations):
            self.optimiser.zero_grad()

            # If multiple functions exist, sample the function from the set of functions.

            if n_functions > 1:
                idx_function = np.random.randint(n_functions)

                x_train = x_trains[idx_function]
                y_train = y_trains[idx_function]

                max_target = x_train.shape[0]

            # During training, we sample n_target points from the function, and
            # randomly select n_context points to condition on.

            num_target = torch.randint(low=5, high=int(0.8*max_target), size=(1,))
            num_context = torch.randint(low=3, high=int(num_target), size=(1,))

            idx = [np.random.permutation(x_train.shape[0])[:num_target] for i in
                   range(batch_size)]
            idx_context = [idx[i][:num_context] for i in range(batch_size)]

            x_target = [x_train[idx[i], :] for i in range(batch_size)]
            y_target = [y_train[idx[i], :] for i in range(batch_size)]
            x_context = [x_train[idx_context[i], :] for i in range(batch_size)]
            y_context = [y_train[idx_context[i], :] for i in range(batch_size)]

            x_target = torch.stack(x_target)
            y_target = torch.stack(y_target)
            x_context = torch.stack(x_context)
            y_context = torch.stack(y_context)

            # The deterministic encoder outputs the deterministic embedding r.
            r = self.det_encoder.forward(x_context, y_context, x_target)  # [batch_size, N_target, r_size]

            # The latent encoder outputs a prior distribution over the
            # latent embedding z (conditioned only on the context points).
            z_priors, _, _ = self.lat_encoder.forward(x_context, y_context)
            z_posteriors, _, _ = self.lat_encoder.forward(x_target, y_target)

            # Sample z from the prior distribution.
            zs = [dist.rsample() for dist in z_priors]
            # [batch_size, r_size]
            z = torch.stack(zs)
            z = z.view(-1, self.r_size)

            # The input to the decoder is the concatenation of the target x values,
            # the deterministic embedding r and the latent variable z
            # the output is the predicted target y for each value of x.
            dists, _, _ = self.decoder.forward(x_target.float(), r.float(), z.float())

            # Calculate the loss
            log_ps = [dist.log_prob(y_target[i, ...].float()) for i, dist in enumerate(dists)]

            log_ps = torch.cat(log_ps, dim=0)

            kl_div = [kl_divergence(z_posterior, z_prior).float() for z_posterior, z_prior
                      in zip(z_posteriors, z_priors)]

            kl_div = torch.stack(kl_div)

            loss = -(torch.mean(log_ps) - torch.mean(kl_div))
            self.losslogger = loss

            # The loss should generally decrease with number of iterations, though it is not
            # guaranteed to decrease monotonically because at each iteration the set of
            # context points changes randomly.
            if (print_freq is not None) and (iteration % print_freq == 500):
                print("Iteration " + str(iteration) + ":, Loss = {:.3f}".format(loss.item()))
            loss.backward()
            self.optimiser.step()

    def predict(self, x_context, y_context, x_target, n_samples=None):
        """
        :param x_context: A tensor of dimensions [1, N_context, x_size].
                          When training N_context is randomly sampled between 3 and N_train;
                          when testing N_context = N_train
        :param y_context: A tensor of dimensions [batch_size, N_context, y_size]
        :param x_target: A tensor of dimensions [N_target, x_size]
        :return dist: The distributions over the predicted outputs y_target
        :return mu: A tensor of dimensionality [batch_size, N_target, output_size]
                    describing the means
                    of the normal distribution.
        :return var: A tensor of dimensionality [batch_size, N_target, output_size]
                     describing the variances of the normal distribution.
        """

        r = self.det_encoder.forward(x_context, y_context, x_target)   # [1, N_target, r_size]
        # The latent encoder outputs a distribution over the latent embedding z.
        dists_z, _, _ = self.lat_encoder.forward(x_context, y_context)   # [1, r_size]

        if n_samples is not None:

            z = dists_z[0].sample((n_samples,)) #[n_samples, r_size]

            z = z.view(-1, self.r_size)

            r = r.repeat(n_samples, 1, 1)
            x_target = x_target.repeat(n_samples, 1, 1)

            # The input to the decoder is the concatenation of the target x values,
            # the deterministic embedding r and the latent variable z
            # the output is the predicted target y for each value of x.
            dists, _, _ = self.decoder.forward(x_target.float(), r.float(), z.float())

            ys = [dist.sample() for dist in dists]

            ys = torch.stack(ys)   #[n_samples, n_target, y_size]

            mu = torch.mean(ys, dim=0)
            var = torch.var(ys, dim=0)

        else:
            zs = [dist.sample() for dist in dists_z]  # [batch_size, r_size]
            z = torch.cat(zs)
            z = z.view(-1, self.r_size)

            # The input to the decoder is the concatenation of the target x values,
            # the deterministic embedding r and the latent variable z
            # the output is the predicted target y for each value of x.
            _, mu, sigma = self.decoder.forward(x_target.float(), r.float(), z.float())
            var = sigma**2

        return mu, var
