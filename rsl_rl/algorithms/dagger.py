import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import numpy as np
from rsl_rl.storage import HistoryStorage 

# computes and returns the latent from the expert
class DaggerExpert(nn.Module):
    def __init__(self, policy, nenvs, geomDim = 0, n_futures = 0):
        super(DaggerExpert, self).__init__()
        self.policy = policy
        self.geomDim = geomDim
        self.n_futures = n_futures
        # mean_pth = loadpth + "/mean" + runid + ".csv"
        # var_pth = loadpth + "/var" + runid + ".csv"
        # obs_mean = np.loadtxt(mean_pth, dtype=np.float32)
        # obs_var = np.loadtxt(var_pth, dtype=np.float32)
        # # cut it
        # obs_mean = obs_mean[:,obs_mean.shape[1]//2:]
        # obs_var = obs_var[:,obs_var.shape[1]//2:]
        # self.mean = self.get_tiled_scales(obs_mean, nenvs, total_obs_size, base_obs_acts_size, T)
        # self.var = self.get_tiled_scales(obs_var, nenvs, total_obs_size, base_obs_acts_size, T)

    # def get_tiled_scales(self, invec, nenvs, total_obs_size, base_obs_acts_size, T):
    #     outvec = np.zeros([nenvs, total_obs_size], dtype = np.float32)
    #     outvec[:, :base_obs_acts_size * T] = np.tile(invec[0, :base_obs_acts_size], [1, T])
    #     outvec[:, base_obs_acts_size * T:] = invec[0]
    #     return outvec

    def forward(self, privilege_obs):
        with torch.no_grad():
            if self.geomDim>0:
                prop_latent = self.policy.prop_encoder(privilege_obs[:, :-self.geomDim*(self.n_futures+1)-1]) # since there is also ref at the end
            else:
                prop_latent = self.policy.prop_encoder(privilege_obs)
            geom_latents = []
            if self.geomDim>0:
                for i in reversed(range(self.n_futures+1)):
                    start = -(i+1)*self.geomDim -1
                    end = -i*self.geomDim -1
                    if (end == 0):
                        end = None
                    geom_latent = self.policy.geom_encoder(privilege_obs[:,start:end])
                    geom_latents.append(geom_latent)
                geom_latents = torch.hstack(geom_latents)
                expert_latent = torch.cat((prop_latent, geom_latents), dim=1)
            else:
                expert_latent = prop_latent
        return expert_latent

class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder, student_mlp,
                 T, history_size, num_obs,device, n_futures=0):
        expert_policy.to(device)
        prop_latent_encoder.to(device)
        #geom_latent_encoder.to(device)
        student_mlp.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder
        #self.geom_latent_encoder = geom_latent_encoder
        self.student_mlp = student_mlp
        self.history_size = history_size
        self.num_obs = num_obs
        self.device = device
        # self.mean = expert_policy.mean
        # self.var = expert_policy.var
        self.n_futures = n_futures
        self.itr = 0
        self.current_prob = 0
        # copy expert weights for mlp policy
        self.student_mlp.architecture.load_state_dict(self.expert_policy.policy.action_mlp.state_dict())


        for net_i in [self.expert_policy.policy, self.student_mlp]:
            for param in net_i.parameters():
                param.requires_grad = False

    def set_itr(self, itr):
        self.itr = itr
        if (itr+1) % 100 == 0:
            self.current_prob += 0.1
            print(f"Probability set to {self.current_prob}")

    def get_history_encoding(self, history):
        # Hack to add velocity
        #velocity = obs[:, self.velocity_idx] -> Velocity thing is not robust
        #raw_obs[:, -3:] = velocity
        prop_latent = self.prop_latent_encoder(history)
        #geom_latent = self.geom_latent_encoder(raw_obs)
        return prop_latent

    def evaluate(self, obs,history):

        prop_latent = self.get_history_encoding(history)
        #expert_latent = self.get_expert_latent(obs)
        #expert_future_geoms = expert_latent[:,prop_latent.shape[1]+geom_latent.shape[1]:]
        # assume that nothing changed
        #geom_latents = []
        #for i in range(self.n_futures + 1):
        #    geom_latents.append(geom_latent)
        #geom_latents = torch.hstack((geom_latent, expert_future_geoms))
        #if np.random.random() < self.current_prob:
        #    # student action
        output = torch.cat([obs, prop_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_expert_action(self, obs, privilege_obs):
        expert_latent = self.get_expert_latent(privilege_obs)
        output = torch.cat([obs, expert_latent], 1)
        #else:
        #    # expert action
        #    output = torch.cat([obs[:, hlen : hlen + obdim], expert_latent], 1)
        output = self.student_mlp.architecture(output)
        return output

    def get_student_action(self, obs,history):
        return self.evaluate(obs,history)

    def get_expert_latent(self, privilege_obs):
        with torch.no_grad():
            latent = self.expert_policy(privilege_obs).detach()
            return latent

    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, example_input, device='cpu'):
        hlen = self.base_obs_acts_size * self.T

        prop_encoder_graph = torch.jit.trace(self.prop_latent_encoder.to(device), example_input[:, :hlen])
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)

        #geom_encoder_graph = torch.jit.trace(self.geom_latent_encoder.to(device), example_input[:, :hlen])
        #torch.jit.save(geom_encoder_graph, fname_geom_encoder)

        mlp_graph = torch.jit.trace(self.student_mlp.architecture.to(device), example_input[:, hlen:])
        torch.jit.save(mlp_graph, fname_mlp)

        self.prop_latent_encoder.to(self.device)
        #self.geom_latent_encoder.to(self.device)
        self.student_mlp.to(self.device)

class DaggerTrainer:
    def __init__(self,
            actor,
            num_envs, 
            num_transitions_per_env,
            obs_shape, latent_shape,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4):

        self.actor = actor
        self.storage = None
        self.optimizer = optim.Adam([*self.actor.prop_latent_encoder.parameters()],
                                    lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.device = device
        self.itr = 0

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def init_storage(self, num_envs, num_transitions_per_env, history_shape, latent_shape, obs_shape):
        self.storage = HistoryStorage(num_envs, num_transitions_per_env, history_shape, latent_shape, self.device)

    def observe(self, history, obs):
        with torch.no_grad():
            actions = self.actor.get_student_action(torch.from_numpy(obs).to(self.device), torch.from_numpy(history).to(self.device))
            #actions = self.actor.get_expert_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def step(self, privilege_obs, history):
        expert_latent = self.actor.get_expert_latent(torch.from_numpy(privilege_obs).to(self.device))
        self.storage.add_inputs(history, expert_latent)

    # def update(self):
    #     # Learning step
    #     mse_loss = self._train_step()
    #     self.storage.clear()
    #     return mse_loss

    def update(self):
        # return loss in the last epoch
        prop_mse = 0
        geom_mse = 0
        loss_counter = 0
        for history_batch, expert_latent_batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):

            predicted_prop_latent = self.actor.get_history_encoding(history_batch)
            loss_prop = self.loss_fn(predicted_prop_latent[:,:8], expert_latent_batch[:,:8])
            # loss_geom = self.loss_fn(predicted_prop_latent[:,8:],
            #         expert_latent_batch[:,8:]) #not implemented
            loss = loss_prop

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            prop_mse += loss_prop.item()
            # geom_mse += loss_geom.item()
            loss_counter += 1
        num_updates = self.num_learning_epochs * self.num_mini_batches
        avg_prop_loss = prop_mse / num_updates 
        # avg_geom_loss = geom_mse / num_updates
        self.storage.clear()
        self.scheduler.step()
        return avg_prop_loss, None
