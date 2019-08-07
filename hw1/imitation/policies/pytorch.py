
import abc
import numpy as np

import torch
from torch import nn
from torch import optim as opt
from torch.functional import F

from tensorboardX import SummaryWriter

from imitation.policies import core

from IPython.core.debugger import set_trace

DEFAULT_SEED = 0
DEFAULT_DEVICE = torch.device( 'cuda:0' if torch.cuda.is_available() else 'cpu' )

class BackbonePytorch( core.Backbone, nn.Module ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( BackbonePytorch, self ).__init__( backboneConfig, **kwargs )

        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE


    def save( self, filepath ) :
        torch.save( self.state_dict(), filepath )


    def load( self, filepath ) :
        self.load_state_dict( torch.load( filepath, map_location = self._device.type ) )


class BackbonePytorchTest( BackbonePytorch ) :

    def __init__( self, backboneConfig, **kwargs ) :
        super( BackbonePytorchTest, self ).__init__( backboneConfig, **kwargs )

        torch.manual_seed( kwargs['seed'] if 'seed' in kwargs else DEFAULT_SEED )

        assert len( self._config.observationsShape ) == 1, 'ERROR> test backbone can\
                only be used with vectorized inputs'
        assert len( self._config.actionsShape ) == 1, 'ERROR> test backbone can\
                only be used with vectorized outputs'

        self._fc1 = torch.nn.Linear( self._config.observationsShape[0], 32 )
        self._fc2 = torch.nn.Linear( 32, 32 )
        self._fc3 = torch.nn.Linear( 32, self._config.actionsShape[0] )


    def forward( self, x ) :
        x = F.relu( self._fc1( x ) )
        x = F.relu( self._fc2( x ) )
        x = self._fc3( x )

        return x


class ModelPytorch( core.Model ) :

    def __init__( self, backbone, **kwargs ) :
        super( ModelPytorch, self ).__init__( backbone, **kwargs )

        self._device = kwargs['device'] if 'device' in kwargs else DEFAULT_DEVICE
        self._optimizer = opt.Adam( self._backbone.parameters(), self._lr )
        self._extension = '.pth'
        self._logger = None
        self._currentTrainLoss = 0.
        self._istep = 0

        self._backbone.to( self._device )


    def predict( self, observation ) :
        # make sure backbone module is in eval mode
        self._backbone.eval()

        with torch.no_grad() :
            _action = self._backbone( torch.from_numpy( observation ).float().to( self._device ) )

        # set backbone back to train mode
        self._backbone.train()

        return _action.cpu().data.numpy()


    def train( self, observations, actions ) :
        actions = torch.from_numpy( actions ).float().to( self._device )
        observations = torch.from_numpy( observations ).float().to( self._device )

        _actionsPred = self._backbone( observations )

        # compute loss (gaussian policy -> mse-loss)
        self._optimizer.zero_grad()
        _loss = F.mse_loss( _actionsPred, actions )
        _loss.backward()

        # update the weights of our backbone
        self._optimizer.step()

        # grab loss value
        self._currentTrainLoss = _loss.item()

        # log training information ---------------------------------------------
        if not self._logger :
            # sanity-check
            assert self._logpath != None, 'ERROR> logpath should be defined'
            # create logger once
            self._logger = SummaryWriter( self._logpath )

        self._logger.add_scalar( 'log_1_loss', self._currentTrainLoss, self._istep )
        # ----------------------------------------------------------------------

        # book keeping
        self._istep += 1


    def logs( self ) :
        msg = 'train-loss = %.5f'

        return msg % ( self._currentTrainLoss )
