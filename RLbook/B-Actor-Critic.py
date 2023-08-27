import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import distrax
import optax
from flax.training.train_state import TrainState
from functools import partial

import gymnax

# --------------------- actor definition + step function -----------------

class actor(nn.Module):
	action_dim: int
	layer_num: int
	layer_size: int

	@nn.compact
	def __call__(self, inputs):
		x = inputs 
		x = nn.relu(nn.Dense(self.layer_size)(x))
		for i in range(self.layer_num):
			x = nn.relu(nn.Dense(self.layer_size)(x))
		x = nn.Dense(self.action_dim)(x)
		return distrax.Categorical(logits=x)

def init_actor(key, action_dim, obs_shape, layer_num, layer_size):

	model = actor(action_dim, layer_num, layer_size)
	model_params = model.init(key, jnp.zeros(obs_shape))

	schedule_fn = optax.linear_schedule(
		init_value=0.001, 
		end_value=0.0005, 
		transition_steps=3000
	)

	tx = optax.adam(schedule_fn)

	actor_ts = TrainState.create(
					apply_fn=model.apply, 
					params=model_params, 
					tx=tx
				)

	return actor_ts

@jax.jit
def step_actor(actor_ts, obs, actions, delta):

	def actor_loss(theta, obs, actions, delta):
		pi = actor_ts.apply_fn(theta, obs)
		log_probs = pi.log_prob(actions)
		return -jnp.sum(log_probs * delta)

	loss_value, grads = jax.value_and_grad(actor_loss,allow_int=True)(
		actor_ts.params, obs, actions, delta
	)
	actor_ts = actor_ts.apply_gradients(grads=grads)
	return loss_value, actor_ts

# ---------------- ------critic definition + step function ----------------

class critic(nn.Module):
	layer_num: int
	layer_size: int

	@nn.compact
	def __call__(self, inputs):
		x = inputs 
		x = nn.relu(nn.Dense(self.layer_size)(x))
		for i in range(self.layer_num):
			x = nn.relu(nn.Dense(self.layer_size)(x))
		x = nn.Dense(1)(x)
		return x

def init_critic(key, obs_shape, layer_num, layer_size):

	model = critic(layer_num, layer_size)
	model_params = model.init(key, jnp.zeros(obs_shape))

	tx = optax.adam(learning_rate=0.1)

	actor_ts = TrainState.create(
					apply_fn=model.apply, 
					params=model_params, 
					tx=tx
				)

	return actor_ts

@partial(jax.jit, static_argnums=4)
def step_critic(state_value_ts, obs, G, delta, mse):

	def book_v_loss(w, obs, G, delta):
		v = jnp.squeeze(state_value_ts.apply_fn(state_value_ts.params, obs))
		return jnp.mean(v * delta)

	def mse_v_loss(w, obs, G, delta):
		# NOTE: delta is unused here but passed to match book_loss
		v = jnp.squeeze(state_value_ts.apply_fn(w, obs))
		return jnp.mean(optax.squared_error(v, G))

	if mse:
		grad_fn = jax.grad(mse_v_loss,allow_int=True)
	else:
		grad_fn = jax.grad(book_v_loss,allow_int=True)

	grads = grad_fn(state_value_ts.params, obs, G, delta)
	state_value_ts = state_value_ts.apply_gradients(grads=grads)
	return state_value_ts

# -------------------------------discounted return + env -------------------------------

@jax.jit
def discounted_returns(rewards, done):

	def cumsum_with_discount(carry, xs):
		new_total_discount =  xs[1] * (0.99 * carry) + xs[0]
		return new_total_discount, new_total_discount 

	carry, returns = jax.lax.scan(cumsum_with_discount, 
									0, 
									[rewards,  (~done).astype(jnp.float32)], 
									reverse=True)
	return returns

@partial(jax.jit, static_argnums=[3,4])
def rollout(rng_input, actor_ts, env_params, steps_in_episode, num_p=32):

    rng_reset, rng_episode = jax.random.split(rng_input)
    reset_rng = jax.random.split(rng_reset, num_p)
    
    #Initialize S (first state of episode) and I = 1
    obs, state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng, env_params)
    # I = jnp.ones(num_p) don't need I since it is batched and we can get discounted_ret

    def policy_step(state_input, tmp):
        obs, state, actor_ts, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        logit = actor_ts.apply_fn(actor_ts.params, obs)
        action = logit.sample(seed=rng_net)
        rng_step = jax.random.split(rng_step, num_p)
        next_obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0,0,0,None))(
            rng_step, state, action, env_params
        )
        carry = [next_obs, next_state, actor_ts, rng]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, actor_ts, rng_episode],
        (),
        steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done

@jax.jit
def compute_delta(critic_ts, rewards, obs):
	new_v = critic_ts.apply_fn(critic_ts.params,obs[1:]).squeeze()
	cur_v = critic_ts.apply_fn(critic_ts.params, obs[:-1]).squeeze()
	return rewards[:-1] + 0.99 * new_v - cur_v 

# -------------- REINFORCE with Baseline (episodic), for estimating pi_θ ≈ pi_* ------------

def train(rng, actor_ts, critic_ts, env_params, EPOCHS, steps_per_episode=500):

	# stats reporting 
	reported_reward, reporting_interval = 0, 100

	# Input: a differentiable policy parameterization pi(a|s, θ) = actor_ts
	# Input: a differentiable state-value function parameterization v(s,w) = critic_ts

	# loop (almost) forever:
	for e in range(EPOCHS):
		# Generate an episode S0,A0,R1, . . . ,ST−1,AT−1,RT , following pi(·|·, θ)
		# S = obs, A = actions, R = rewards, and done is a mask for episode ends
		# code is in .utils, and is batched with num_p across processes on GPU
		obs, actions, rewards, _, done = rollout(
			rng, actor_ts, env_params, steps_per_episode+1, num_p=64
		)

		delta = 0
		G = jax.vmap(discounted_returns, in_axes=1)(rewards, done).T
		critic_ts = step_critic(critic_ts, obs, G, delta, mse=True)

		# Loop for each step of the episode t = 0, 1, . . . ,T − 1:
		# update is batched for faster and better
		# discount delta??
		delta = compute_delta(critic_ts, rewards, obs)
		loss_value, actor_ts = step_actor(actor_ts, obs[:-1], actions[:-1], delta)


		# stats reporting
		reported_reward += jnp.sum(rewards) / max(jnp.sum(done.astype(jnp.float32)),1) 

		if e % reporting_interval == 0: 
			print(f"Epoch: {e}, Average reward = {reported_reward / reporting_interval}")
			reported_reward = 0 

	print(f"Epoch: {e}, Average reward = {reported_reward / reporting_interval}")
	return actor_ts, critic_ts

#------------------------------------------------------------------------------------------

env, env_params = gymnax.make("CartPole-v1")

num_actions = env.num_actions
num_obs = env.observation_space(env_params).shape

rng = random.PRNGKey(491495711086)
key_actor, key_critic, rng = random.split(rng, 3)

actor_ts = init_actor(key_actor, num_actions, num_obs, 1, 16)
critic_ts = init_critic(key_critic, num_obs, 1, 16)

actor_ts, critic_ts = train(rng, actor_ts, critic_ts, env_params, 5000, 500)
