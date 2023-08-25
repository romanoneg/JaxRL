import jax
from jax import lax, random, numpy as jnp
import flax
from flax import linen as nn
import distrax
import optax
from flax.training.train_state import TrainState
from functools import partial

import gymnax

# --------------------- policy definition + step function -----------------

class policy(nn.Module):
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

def init_policy(key, action_dim, obs_shape, layer_num, layer_size):

	model = policy(action_dim, layer_num, layer_size)
	model_params = model.init(key, jnp.zeros(obs_shape))

	schedule_fn = optax.linear_schedule(
		init_value=0.001, 
		end_value=0.0005, 
		transition_steps=3000
	)

	tx = optax.adam(schedule_fn)

	train_state = TrainState.create(
					apply_fn=model.apply, 
					params=model_params, 
					tx=tx
				)

	return train_state

@jax.jit
def step_policy(policy_ts, obs, actions, delta):

	def policy_loss(theta, obs, actions, delta):
		pi = policy_ts.apply_fn(theta, obs)
		log_probs = pi.log_prob(actions)
		return -jnp.sum(log_probs * delta)

	loss_value, grads = jax.value_and_grad(policy_loss,allow_int=True)(
		policy_ts.params, obs, actions, delta
	)
	policy_ts = policy_ts.apply_gradients(grads=grads)
	return loss_value, policy_ts

# ---------------- state_value network definition + step function ------------

class state_value(nn.Module):
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

def init_state_value(key, obs_shape, layer_num, layer_size):

	model = state_value(layer_num, layer_size)
	model_params = model.init(key, jnp.zeros(obs_shape))

	tx = optax.adam(learning_rate=0.1)

	train_state = TrainState.create(
					apply_fn=model.apply, 
					params=model_params, 
					tx=tx
				)

	return train_state

@partial(jax.jit, static_argnums=4)
def step_state_value(state_value_ts, obs, G, delta, mse):

	def book_v_loss(w, obs, G, delta):
		v = jnp.squeeze(state_value_ts.apply_fn(state_value_ts.params, obs))
		return jnp.mean(v * delta)

	def mse_v_loss(w, obs, G, delta):
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
def rollout(rng_input, train_state, env_params, steps_in_episode, num_p=32):
    """Rollout a jitted gymnax episode with lax.scan."""
    # from the gymnax getting started guide, with some changes
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    reset_rng = jax.random.split(rng_reset, num_p)
    # CHANGED HERE TO ADD VMAP - batch the environment reset
    obs, state = jax.vmap(env.reset, in_axes=(0,None))(reset_rng, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, train_state, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        logit = train_state.apply_fn(train_state.params, obs)
        action = logit.sample(seed=rng_net)
        rng_step = jax.random.split(rng_step, num_p)
        # CHANGED HERE TO ADD VMAP - batch the environment step, returns batched data
        next_obs, next_state, reward, done, _ = jax.vmap(env.step, in_axes=(0,0,0,None))(
            rng_step, state, action, env_params
        )
        carry = [next_obs, next_state, train_state, rng]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, train_state, rng_episode],
        (),
        steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done

# -------------- REINFORCE with Baseline (episodic), for estimating pi_θ ≈ pi_* ------------

def train(rng, policy_ts, state_value_ts, env_params, EPOCHS, steps_per_episode=500):

	# stats reporting 
	reported_reward, reporting_interval = 0, 100

	# Input: a differentiable policy parameterization pi(a|s, θ) = policy_ts
	# Input: a differentiable state-value function parameterization v(s,w) = state_value_ts

	# loop (almost) forever:
	for e in range(EPOCHS):
		# Generate an episode S0,A0,R1, . . . ,ST−1,AT−1,RT , following pi(·|·, θ)
		# S = obs, A = actions, R = rewards, and done is a mask for episode ends
		# code is in .utils, and is batched with num_p across processes on GPU
		obs, actions, rewards, _, done = rollout(
			rng, policy_ts, env_params, steps_per_episode, num_p=64
		)

		# Loop for each step of the episode t = 0, 1, . . . ,T − 1:
		# update is batched for faster and better
		G = jax.vmap(discounted_returns, in_axes=1)(rewards, done).T
		delta = G - jnp.squeeze(state_value_ts.apply_fn(state_value_ts.params, obs))
		state_value_ts = step_state_value(state_value_ts, obs, G, delta, mse=True)
		loss_value, policy_ts = step_policy(policy_ts, obs, actions, delta)


		# stats reporting
		reported_reward += jnp.sum(rewards) / jnp.sum(done.astype(jnp.float32)) 

		if e % reporting_interval == 0: 
			print(f"Epoch: {e}, Average reward = {reported_reward / reporting_interval}")
			reported_reward = 0 

	print(f"Epoch: {e}, Average reward = {reported_reward / reporting_interval}")
	return policy_ts, state_value_ts

#------------------------------------------------------------------------------------------

env, env_params = gymnax.make("CartPole-v1")

num_actions = env.num_actions
num_obs = env.observation_space(env_params).shape

rng = random.PRNGKey(491495711086)
key_policy, key_state_value, rng = random.split(rng, 3)

policy_ts = init_policy(key_policy, num_actions, num_obs, 1, 16)
state_value_ts = init_state_value(key_state_value, num_obs, 1, 16)

policy_ts, state_value_ts = train(rng, policy_ts, state_value_ts, env_params, 5000, 500)
