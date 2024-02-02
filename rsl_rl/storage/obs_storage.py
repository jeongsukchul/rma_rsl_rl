import torch

class ObsStorage:
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, latent_shape, device):
        self.device = device

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape).to(self.device)
        self.expert_latents = torch.zeros(num_transitions_per_env, num_envs, *latent_shape).to(self.device)
        self.device = device

        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.step = 0

    def add_obs(self, obs, expert_action):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(torch.from_numpy(obs).to(self.device))
        self.expert_latents[self.step].copy_(expert_action)
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        expert_latents = self.expert_latents.flatten(0,1)
        for epoch in range(num_epochs):
            for batch_id in range(num_mini_batches):

                start = batch_id*mini_batch_size
                end = (batch_id+1)*mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                expert_latents_batch = expert_latents[batch_idx]
                yield obs_batch, expert_latents_batch