import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.modules import ActorCriticLatent
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms import PPO

# This algorithm includes the mirror loss
# https://arxiv.org/pdf/1801.08093.pdf

class PPO_sym(PPO):
    actor_critic: ActorCriticLatent
    def __init__(self, 
                 actor_critic, 
                 mirror, mirror_neg = {},
                 no_mirror = 12, 
                 mirror_weight = 4, 
                 num_learning_epochs=1, 
                 num_mini_batches=1, 
                 clip_param=0.2, 
                 gamma=0.998, 
                 lam=0.95, 
                 value_loss_coef=1, 
                 entropy_coef=0, 
                 learning_rate=0.001, 
                 max_grad_norm=1, 
                 use_clipped_value_loss=True, 
                 schedule="fixed", 
                 desired_kl=0.01, 
                 device='cpu'):
        super().__init__(actor_critic, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl, device)
        self.mirror_dict = mirror
        self.mirror_neg_dict = mirror_neg
        self.no_mirror = no_mirror
        self.mirror_weight = mirror_weight
        self.mirror_init = True
        print("Sym version of PPO loaded")
        print("Mirror weight: ", mirror_weight)
                
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_mirror_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:
            # The shape of an obs batch is : (minibatchsize, obs_shape)

                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate
                
                # Mirror loss
                # use mirror dict as mirror
                if self.mirror_init:
                    num_obvs = obs_batch.shape[1] # 75
                    minibatchsize = obs_batch.shape[0]
                    num_acts = actions_batch.shape[1] # 21
                    self.mirror_obs = torch.eye(num_obvs).reshape(1, num_obvs, num_obvs).repeat(minibatchsize, 1, 1).to(device=self.device)
                    self.mirror_act = torch.eye(num_acts).reshape(1, num_acts, num_acts).repeat(minibatchsize, 1, 1).to(device=self.device)
                    for _, (i,j) in self.mirror_dict.items():
                        self.mirror_act[:, i, i] = 0
                        self.mirror_act[:, j, j] = 0
                        self.mirror_act[:, i, j] = 1
                        self.mirror_act[:, j, i] = 1
                    for _, (i, j) in self.mirror_neg_dict.items():
                        self.mirror_act[:, i, i] = 0
                        self.mirror_act[:, j, j] = 0
                        self.mirror_act[:, i, j] = -1
                        self.mirror_act[:, j, i] = -1
                    for i in range(int(self.no_mirror / 3)):
                        if (i == 1): # base angular velocity terms -> *-1 to x and z ang vels
                            self.mirror_obs[:, 3*i, 3*i] *= -1
                            self.mirror_obs[:, 3*i+2, 3*i+2] *= -1 
                        if (i == 3):
                            self.mirror_obs[:, 3*i+1, 3*i+1] *= -1
                            self.mirror_obs[:, 3*i+2, 3*i+2] *= -1 # last element of command is yaw
                        else:
                            self.mirror_obs[:, 3*i+1, 3*i+1] *= -1
                    for i in range(int((num_obvs - self.no_mirror) / num_acts)):
                        self.mirror_obs[:, self.no_mirror + i*num_acts:self.no_mirror + (i+1)*num_acts, self.no_mirror + i*num_acts:self.no_mirror + (i+1)*num_acts] = self.mirror_act
                    self.mirror_init = False
                mirror_loss = torch.mean(torch.square(self.actor_critic.actor(obs_batch) \
                                                      - (self.mirror_act @ self.actor_critic.actor((self.mirror_obs @ obs_batch.unsqueeze(2)).squeeze()).unsqueeze(2)).squeeze())) 
            
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = self.mirror_weight * mirror_loss +surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_mirror_loss += mirror_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_mirror_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_mirror_loss