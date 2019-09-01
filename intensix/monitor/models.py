"""Recurrent models for time series prediction.
"""

import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def make(arch, input_size, **args):
    """Creates a model object based on the model architecture,
    as appears in the configuration file. Facilitates creation
    of model objects from stored models.
    """
    # An entry for each achitecture must be added
    # to the dictionary
    return {"rnn": P_RNN,
            "mrnn": P_RNN,  # deprecated, kept for compatibility
            "grnn": P_GRNN,
            }[arch](input_size, **args)


class P(nn.Module):
    """Generic sequence predictor.
    Accepts input_size as the parameter, and for each time step
    outputs depth predictions. Each prediction is a vector of
    size 2*input_size:
      means o variances
    followed by standard deviations.
    """
    def __init__(self, input_size):
        """Initializes P. For forward propagation,
        the input must be twice the input size,
        deviations following means.
        """
        super(P, self).__init__()
        self.input_size = input_size

    def makex(self, batch, depth):
        """Expands a batch into input (x).
        Arguments:
          batch -- data batch.
          depth -- prediction depth.
        """
        # depth can be 0
        means = batch[:batch.size(0) - depth]
        stds = means.clone()
        stds.zero_()
        x = torch.cat([means, stds], dim=-1)
        return x

    def makey(self, batch, depth):
        """Expands a batch into ground truth (y).
        Arguments:
          batch -- data batch.
          depth -- prediction depth.
        """
        assert depth > 0, "depth must be positive"
        ylayers = [batch[i + 1:batch.size(0) - depth + i + 1]
                   for i in range(depth)]
        y = torch.stack(ylayers, dim=2)
        return y

    def makexy(self, batch, depth):
        """Expands a batch into input (x) and ground truth (y).
        Arguments:
          batch -- data batch.
          depth -- prediction depth.

        The input is augmented by zero standard deviation for
        each component and shortened (along the time line) by
        the depth. E.g. [[1], [-0.5], [0.3], [-0.3]]
        for depth=2 becomes [[1, 0], [-0.5, 0]].

        The truth is layered so that there are depth
        points in the future for each time step. E.g. for the
        same input the truth is [[[-0.5], [0.3]], [[0.3], [-0.3]]].

        The second dimension is the batch dimension, not
        shown in the examples. See the unit tests for more
        detailed examples.
        """
        x = self.makex(batch, depth)
        y = self.makey(batch, depth)
        return x, y

    def pred_nlls(self, preds, y, eps=1E-6):
        """Computes prediction NLL of y relative to predictions.
        Returns a tensor of nlls.
        """
        # eps is to avoid division by zero
        mean = preds[:, :, :, :self.input_size]
        std = preds[:, :, :, self.input_size:]
        dy = y - mean
        var = std * std + eps

        # Log-likelihood for normal distribution,
        # with constant term and factor omitted.
        nlls = torch.log(var) + dy * dy / var

        return nlls

    def initial_xo(self, x):
        """Creates xo (the predicted output) before the first
        input. Used when missing inputs must be filled.
        """
        xo = x[0].clone()
        # initialize xo to the prior in order to handle
        # missing observations
        xo[:, :self.input_size] = 0.  # zero mean
        xo[:, self.input_size:] = 1.  # unit std
        return xo

    def initial_h(self, x):
        """Creates the initial hidden state.
        """
        h = Variable(
            torch.zeros(self.nlayers, x.size(1), self.hidden_size))
        if isinstance(x.data, torch.cuda.FloatTensor):
            h = h.cuda()
        return h

    def fill_missing(self, xt, xo):
        """Fills missing observations in xt with predictions
        from xo. Returns updated xt.
        """
        # PyTorch does not provide isnan, need to fall back to
        # numpy here; alternatively, nans should be replaced
        # in the data with a reserved value

        # compute nan indices for means
        nans = numpy.isnan(xt.data.numpy()[:, :self.input_size])
        if numpy.any(nans):
            nans = torch.from_numpy(nans.astype(int)).byte()
            # extend the index array to stds
            nans = Variable(torch.cat([nans, nans], dim=1))

            # make an updated copy of xt
            xt = xt.clone()
            xt.masked_scatter_(nans, xo.masked_select(nans))
        return xt


class P_RNN(P):
    """Sequence predictor based on multilayer-layer GRU.
    """
    def __init__(self, input_size, hidden_size, nlayers, p):
        """Creates an instance of rnn model. Arguments:
          input_size -- the number of input features,
          hidden_size -- the hidden size of a GRU cell,
          nlayers -- number of RNN layers,
          p -- dropout probability.
        """
        super(P_RNN, self).__init__(input_size)
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.p = p
        self.x_size = 2 * input_size   # means and stds

        self.grus = nn.ModuleList(
                [nn.GRUCell(self.x_size, self.hidden_size)] +
                [nn.GRUCell(self.hidden_size, self.hidden_size)
                 for i in range(nlayers - 1)])
        self.dropout = nn.Dropout(p=self.p)

        # Readout --- MLP with linear last layer
        self.readout = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # We allow negative stds on output
            # and just use the absolute value
            # for a simpler code
            nn.Linear(hidden_size, self.x_size))

    def rnncell(self, x, h):
        """Multilayer GRU. Arguments:
          x -- input
          h -- hidden state
        Returns next hidden state.
        """
        hnext = h.clone()
        z = x
        for i in range(self.nlayers - 1):
            hnext[i] = self.grus[i](z, h[i])
            z = self.dropout(hnext[i])
        hnext[-1] = self.grus[-1](z, h[-1])

        return hnext

    def step(self, xt, depth, xo, h, missing):
        """Performs a single step through RNN. Arguments:
          xt -- the step input,
          depth -- prediction depth,
          xo -- current predictions,
          h  -- the hidden state,
          missing -- if True, fill missing values.
        Returns next prediction, hidden state, future
        predictions up to depth.
        """
        if missing:
            # fill missing observations from predictions
            xt = self.fill_missing(xt, xo)
        # Update the real state
        h = self.rnncell(xt, h)
        # Generate predictions of depth future time steps
        # as seen from time t
        xo = self.readout(h[-1])
        xos = [xo]
        # The prediction state branches out of
        # real state and is forgotten after each
        # real time step
        xoo = xo
        hh = h
        for _ in range(depth - 1):
            hh = self.rnncell(xoo, hh)
            xoo = self.readout(hh[-1])
            xos.append(xoo)
        xos = torch.stack(xos, 1)
        return xo, h, xos

    def forward(self, x, depth, xo=None, h=None, missing=False):
        """Performs the forward pass over the network.
        Arguments:
          x -- the observations.
          depth -- prediction depth.
          xo -- the initial predictions; initialized to the
                prior unless provided.
          h --  the initial hidden stat; initialized to zeroes
                unless provided.
          missing -- if True, replaces missing observations
                     with predictions.
        Returns predicted outputs and the last hidden state.
        """
        if missing and xo is None:
            xo = self.initial_xo(x)
        if h is None:
            h = self.initial_h(x)
        preds = []
        for t in range(x.size(0)):
            xt = x[t]
            xo, h, xos = self.step(xt, depth, xo, h, missing)
            preds.append(xos)
        preds = torch.stack(preds, 0)

        return preds, h

    def loss(self, result, x, y):
        """Computes model loss as average NLL over all predictions.
        Arguments:
          result -- result of the forward pass,
          x -- input,
          y -- output.
        Returns the loss as a Variable.
        """
        preds, _ = result
        nlls = self.pred_nlls(preds, y)
        loss = nlls.mean()
        return loss


# Discriminator labels
_ENCODED_LABEL = 0
_SAMPLED_LABEL = 1


class P_GRNN(P_RNN):
    """Sequence predictor based on multilayer GRU with
    componentwise noise gaussianization.
    """
    def __init__(self, input_size, hidden_size,
                 nlayers, p,
                 encoder_sizes, discriminator_sizes):
        """Creates an instance of grnn model. Arguments:
          input_size -- the number of input features,
          hidden_size -- the hidden size of a GRU cell,
          nlayers -- number of RNN layers,
          p -- dropout probability,
          encoder_sizes -- sizes of encoder's hidden layers,
          discriminator_sizes -- sizes of discriminator's
                                 hidden layers.
        """
        super(P_GRNN, self).__init__(input_size, hidden_size,
                                     nlayers, p)

        # We need a separate encoder for each input dimension
        self.encoders = nn.ModuleList([self.__make_encoder(
                                           encoder_sizes)
                                       for _ in range(input_size)])

        # Discriminators are much easier to train if they are
        # separate too
        self.discriminators = nn.ModuleList([
            self.__make_discriminator(discriminator_sizes)
            for _ in range(input_size)])

    def __make_encoder(self, encoder_sizes):
        """Creates an encoder module. The input is
        x, mu, sigma, the output is x. Arguments:
          encoder_sizes --- sizes of module's hidden layers.
        Returns the module.
        """
        layers = sum([[nn.Linear(si, so),
                       nn.ReLU()]
                      for si, so in zip([3] + encoder_sizes[:-1],
                                        encoder_sizes)],
                     [])
        # we do not need the last activation, the output is unrestricted
        layers.append(nn.Linear(encoder_sizes[-1], 1))
        return nn.Sequential(*layers)

    def __make_discriminator(self, discriminator_sizes):
        """Creates a discriminator module. The input is x,
        the output is [0, 1]. Arguments:
          encoder_sizes --- sizes of module's hidden layers.
        Returns the module.
        """
        layers = sum([[nn.Linear(si, so),
                       nn.ReLU()]
                      for si, so in zip([1] + discriminator_sizes[:-1],
                                        discriminator_sizes)],
                     [])
        # the last activation must be sigmoid rather than ReLU
        layers.extend([nn.Linear(discriminator_sizes[-1], 1),
                       nn.Sigmoid()])
        return nn.Sequential(*layers)

    def encode(self, xt, xo, missing):
        """Encodes xt according to xo.
        """
        # augment xt with means and stds
        xms = torch.stack([xt[:, :self.input_size],
                           xo[:, :self.input_size],  # means
                           xo[:, self.input_size:]   # stds
                           ], dim=2)

        # encode each component
        xr = xt.clone()
        for i in range(self.input_size):
            xr[:, [i]] = self.encoders[i](xms[:, i, :])

        if missing:
            # put NaNs back, we should not need it as NaN is very
            # contagious, but just for safety
            nans = numpy.isnan(xt.data.numpy())
            if numpy.any(nans):
                nans = torch.from_numpy(nans.astype(int)).byte()
                xr.data.masked_fill_(nans, numpy.nan)

        return xr

    def forward(self, x, depth, xo=None, h=None, missing=False):
        """Performs the forward pass over the network. Arguments:
          x -- the observations.
          depth -- prediction depth.
          xo -- the initial predictions; initialized to the
                prior unless provided.
          h --  the initial hidden stat; initialized to zeroes
                unless provided.
          missing -- if True, replaces missing observations
                     with predictions.
        """
        if xo is None:
            xo = self.initial_xo(x)
        if h is None:
            h = self.initial_h(x)
        preds = []

        # For loss computation
        encoded = []  # encoded and normalized input
        for t in range(x.size(0)):
            xt = x[t]
            xe = self.encode(xt, xo, missing)

            # Accumulate encoded input to compute the loss
            encoded.append(xe)

            xo, h, xos = self.step(xe, depth, xo, h, missing)
            preds.append(xos)

        preds = torch.stack(preds, 0)
        encoded = torch.stack(encoded, 0)
        return preds, h, encoded

    def loss(self, result, x, y):
        """Computes model loss as average NLL over all predictions.
        Arguments:
          result -- result of the forward pass,
          x -- input,
          y -- output.
        Returns the loss as a Variable.
        """
        preds, _, encoded = result

        # Each dimension is gaussianized independently
        encoded = encoded[:, :, :self.input_size]
        encoded = encoded.resize(encoded.numel()//self.input_size,
                                 self.input_size)

        # Samples from normal distribution for training
        # the discriminator
        sampled = Variable(encoded.data.clone())
        sampled.data.normal_(0., 1.)

        # Labels for training
        encoded_labels = Variable(encoded.data[:, [0]].clone())
        encoded_labels.data.fill_(_ENCODED_LABEL)
        sampled_labels = encoded_labels.clone()
        sampled_labels.data.fill_(_SAMPLED_LABEL)

        # Train the discriminator
        sampled = encoded.detach().clone()
        sampled.data.normal_(0., 1.)

        # Computing discriminator loss
        dsc_loss = 0.
        for i in range(self.input_size):
            dsc_loss += F.binary_cross_entropy(
                self.discriminators[i](encoded[:, [i]].detach()),
                encoded_labels)
            dsc_loss += F.binary_cross_entropy(
                self.discriminators[i](sampled[:, [i]]),
                sampled_labels)
        dsc_loss = dsc_loss / self.input_size / 2 - numpy.log(2)

        # Prediction loss
        pred_loss = self.pred_nlls(preds, y).mean()
        # Perplexion loss
        prp_loss = 0.
        for i in range(self.input_size):
            # Perplexer is a temporary module to compute the loss
            # without updating discriminator weights
            perplexer = copy.deepcopy(self.discriminators[i])
            prp_loss += F.binary_cross_entropy(
                perplexer(encoded[:, [i]]),
                sampled_labels)
        prp_loss = prp_loss / self.input_size - numpy.log(2)

        print("\t\t\t\t\t[dsc: {:.4f} prp: {:.4f} pred: {:.4f}]"
              .format(dsc_loss.data[0], prp_loss.data[0],
                      pred_loss.data[0]),
              end="\r")

        return dsc_loss + pred_loss + prp_loss


def compute_kls(p, q):
    """Computes KL distances for each point between two predictions.
    Arguments --- two three-dimensional tensors (predictions).
    Returns --- three-dimensional tensor of KL distances.
    """
    input_size = p.size(2) // 2
    mp = p[:, :, :input_size]
    mq = q[:, :, :input_size]
    sp = torch.abs(p[:, :, input_size:])
    sq = torch.abs(q[:, :, input_size:])
    dpq = mp - mq
    kls = -0.5 + torch.log(sq / sp) + (sp * sp + dpq * dpq) / \
        (2 * sq * sq)
    return kls


class Alert(nn.Module):
    """Alert filter model. Accepts alert descriptor
    as created based on change points and classifies
    as either relevant (1) or irrelevant (0) alert.
    """

    def __init__(self, hidden_size, p=0.5):
        super(Alert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        """Forward pass on alert filter. Takes
        the alert description in the form of NLL,
        entry state, exit state, and returns the
        probability of relevance.
        """
        return self.mlp(x)

    def loss(self, z, y, weight):
        """Computes classification loss.
        """
        return F.binary_cross_entropy(z, y, weight)
