import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal

class MLPEncode(nn.Module):
    def __init__(self, shape,
                 actionvation_fn, 
                 base_obdim, 
                 output_size, 
                 output_activation_fn = None, 
                 small_init= False, 
                 priv_dim = 261, 
                 geom_dim = 0,
                 n_futures = 0):
        super(MLPEncode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        self.base_obs_dim = base_obdim
        self.geom_dim = geom_dim
        
        ## Encoder Architecture
        prop_latent_dim = 8
        if self.geom_dim >0:
            geom_latent_dim = 1
        else:
            geom_latent_dim = 0
        self.n_futures = n_futures
        self.prop_latent_dim = prop_latent_dim
        self.geom_latent_dim = geom_latent_dim
        print("priv_dim",priv_dim)
        print(self.activation_fn)
        self.prop_encoder =  nn.Sequential(*[
                                    nn.Linear(priv_dim, 256), self.activation_fn,
                                    nn.Linear(256, 128), self.activation_fn,
                                    nn.Linear(128, prop_latent_dim), self.activation_fn,
                                    ]) 
        if self.geom_dim > 0:
            self.geom_encoder =  nn.Sequential(*[
                                        nn.Linear(geom_dim, 64), self.activation_fn,
                                        nn.Linear(64, 16), self.activation_fn,
                                        nn.Linear(16, geom_latent_dim), self.activation_fn,
                                        ]) 
        else:
            self.geom_encoder = None
        scale_encoder = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        # creating the action encoder
        modules = [nn.Linear(self.base_obs_dim + prop_latent_dim + (self.n_futures + 1)*geom_latent_dim, shape[0]), self.activation_fn]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn)
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn)
        self.action_mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.action_mlp, scale)
        self.init_weights(self.prop_encoder, scale_encoder)
        if self.geom_dim >0:
            self.init_weights(self.geom_encoder, scale_encoder)
        if small_init: action_output_layer.weight.data *= 1e-6

        self.input_shape = [base_obdim]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

    def forward(self, x):
        # get only the x related to the control policy
        # if x.shape[1] > 130:
        #     # Hacky way to detect where you are (dagger or not)
        #     # TODO: improve on this!
        #     x = x[:,x.shape[1]//2:]
        if self.geom_dim>0:
            prop_latent = self.prop_encoder(x[:,self.base_obs_dim:-self.geom_dim*(self.n_futures+1) -1])
        else:
            prop_latent = self.prop_encoder(x[:,self.base_obs_dim:])

        # geom_latents = []
        # if self.geom_dim>0:
        #     for i in reversed(range(self.n_futures+1)):
        #         start = -(i+1)*self.geom_dim -1
        #         end = -i*self.geom_dim -1
        #         if (end == 0): 
        #             end = None
                
        #             geom_latent = self.geom_encoder(x[:,start:end])
        #             geom_latents.append(geom_latent)
        #     geom_latents = torch.hstack(geom_latents)
        #     input = torch.cat([x[:,:self.base_obs_dim], prop_latent, geom_latents], 1)

        input = torch.cat([x[:,:self.base_obs_dim], prop_latent], 1)
        return self.action_mlp(input)
    def only_obs(self,x):
        return self.action_mlp(x)

class MLPEncode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, priv_dim = 261, geom_dim = 0, n_futures = 0):
        super(MLPEncode_wrap, self).__init__()
        self.architecture = MLPEncode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, priv_dim, geom_dim, n_futures)
        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class ActorCriticLatent(nn.Module):
    is_recurrent = False
    def __init__(self,  num_obs,
                        privDim, 
                        geomDim,
                        n_futures,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        output_activatation='tanh',
                        init_noise_std=1.0, 
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCriticLatent, self).__init__()

        activation = get_activation(activation)

        self.actor = MLPEncode_wrap(actor_hidden_dims,
                                     activation,
                                     num_obs,
                                     num_actions,
                                     priv_dim = privDim,
                                     geom_dim = geomDim,
                                     n_futures = n_futures,
                                     output_activation_fn=output_activatation)
        


        self.critic = MLPEncode_wrap(critic_hidden_dims,
                                     activation,
                                     num_obs,
                                     1,
                                     priv_dim = privDim,
                                     geom_dim = geomDim,
                                     n_futures = n_futures,
                                     output_activation_fn=output_activatation)
        # Value function

        print(f"Actor MLP: {self.actor.architecture}")
        print(f"Critic MLP: {self.critic.architecture}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor.architecture(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor.architecture(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic.architecture(critic_observations)
        return value

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
