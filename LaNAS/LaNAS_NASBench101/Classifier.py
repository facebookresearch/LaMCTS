# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 
import torch
import torch.nn as nn
from torch.autograd import Variable
import json
from torch import optim
import numpy as np

# this is the backbone model
# to split networks at a MCTS state
class LinearModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_( self.fc1.weight )
    
    def forward(self, x):
        y = self.fc1(x)
        #print("=====>X_shape:", x.shape)
        return y

# the input will be samples!
class Classifier():
    def __init__(self, samples, input_dim):
        self.training_counter = 0
        assert input_dim >= 1
        assert type(samples) ==  type({})
        self.input_dim  = input_dim
        self.samples    = samples
        self.model      = LinearModel(input_dim, 1)

        if torch.cuda.is_available():
            self.model.cuda()
        self.l_rate     = 0.00001
        self.optimiser  = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs     = 1 #TODO:revise to 100
        self.boundary   = -1
        self.nets       = []
        
    def get_params(self):
        return self.model.fc1.weight.detach().numpy(), self.model.fc1.bias.detach().numpy()

    def reinit(self):
        torch.nn.init.xavier_uniform_( self.m.fc1.weight )
        torch.nn.init.xavier_uniform_( self.m.fc2.weight )
    
    def update_samples(self, latest_samples):
        assert type(latest_samples) == type(self.samples)
        sampled_nets    = []
        nets_acc        = []
        for k, v in latest_samples.items():
            net = json.loads(k)
            sampled_nets.append( net )
            nets_acc.append( v )
        self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32).reshape(-1, self.input_dim))
        self.acc  = torch.from_numpy(np.asarray(nets_acc,     dtype=np.float32).reshape(-1, 1))
        self.samples = latest_samples
        if torch.cuda.is_available():
            self.nets = self.nets.cuda()
            self.acc  = self.acc.cuda()

    def train(self):
        if self.training_counter == 0:
            self.epochs = 20000
        else:
            self.epochs = 3000
        self.training_counter += 1
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        for epoch in range(self.epochs):
            epoch += 1
            nets = self.nets
            acc  = self.acc
            #clear grads
            self.optimiser.zero_grad()
            #forward to get predicted values
            outputs = self.model.forward( nets )
            loss = nn.MSELoss()(outputs, acc)
            loss.backward()# back props
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimiser.step()# update the parameters
#            if epoch % 1000 == 0:
#                print('@' + self.__class__.__name__ + ' epoch {}, loss {}'.format(epoch, loss.data))

    def predict(self, remaining):
        assert type(remaining) == type({})
        remaining_archs    = []
        for k, v in remaining.items():
            net = json.loads(k)
            remaining_archs.append( net )
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32).reshape(-1, self.input_dim))
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()
        outputs = self.model.forward(remaining_archs)
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            outputs         = outputs.cpu()
        result  = {}
        counter = 0
        for k in range(0, len(remaining_archs) ):
            counter += 1
            arch = remaining_archs[k].detach().numpy()
            arch_str = json.dumps( arch.tolist() )
            result[ arch_str ] = outputs[k].detach().numpy().tolist()[0]
        assert len(result) == len(remaining)
        return result

    def split_predictions(self, remaining):
        assert type(remaining) == type({})
        samples_badness  = {}
        samples_goodies  = {}
        if len(remaining) == 0:
            return samples_badness, samples_goodies
        predictions = self.predict(remaining)
        avg_acc          = self.predict_mean()
        self.boundary    = avg_acc
        for k, v in predictions.items():
            if v < avg_acc:
                samples_badness[k] = v
            else:
                samples_goodies[k] = v
        assert len(samples_badness) + len(samples_goodies) == len(remaining)
        return  samples_goodies, samples_badness


    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual acc?
        outputs    = self.model.forward(self.nets)
        pred_np    = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    
    def split_data(self):
        samples_badness  = {}
        samples_goodies  = {}
        if len(self.nets) == 0:
            return samples_badness, samples_goodies
        self.train()
        avg_acc          = self.predict_mean()
        self.boundary    = avg_acc
        for k, v in self.samples.items():
            if v < avg_acc:
                samples_badness[k]  = v
            else:
                samples_goodies[k] = v
        assert len(samples_badness) + len(samples_goodies) == len( self.samples )
        return  samples_goodies, samples_badness
