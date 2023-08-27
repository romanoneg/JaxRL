# JaxRL - romanoneg

![image](https://github.com/romanoneg/JaxRL/assets/43445765/3a41737e-4be5-4576-83b6-7ea76e25dd60)

---
### First Tests, REINFORCE agents

As a very quick preliminary test here are the times each script took to solve `CartPole-v1` :

| Script        | Time     | % change | Notes |
|--------------|-----------|------------|-----------|
| [REINFORCE_pytorch](https://github.com/romanoneg/JaxRL/blob/main/REINFORCE/REINFORCE_pytorch.py)   | 9m12s   | -- | by far the easiest to pickup        |
| [REINFORCE_jax](https://github.com/romanoneg/JaxRL/blob/main/REINFORCE/REINFORCE_jax.py) | 58.6s  | ~90% faster | includes compile time       |
| [vmap_REINFORCE_jax](https://github.com/romanoneg/JaxRL/blob/main/REINFORCE/vmap_REINFORCE_jax.py)     | 🔥**40.0s**🔥 | ~93% faster | include compile time       |
| pmap_vmap_REINFORCE_jax| --            | Prob really fast| my second GPU is too old to test 🙃  |

(Jax Scripts use the [gymnax](https://github.com/RobertTLange/gymnax) library)

### Speed metrics:

Using `vmap_REINFORCE_jax.py` script I was able to run `CartPole-v1` at `0.055s/1M` step transitions while training the\
network on an A100 with 2k envs:

> 55.7 ms ± 1.05 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
  
A little bit faster than the baseline of `0.08s/1M` step transistions given in [gymnax](https://github.com/RobertTLange/gymnax).

---
### Reinforcement Learning Book by Sutton & Barto - Chapter 13: Policy Gradient 

Focused on recreating each of the algorithms talked about in the chapter in jax (code often gets transfered in the following).

- [REINFORCE_jax](https://github.com/romanoneg/JaxRL/blob/main/REINFORCE/REINFORCE_jax.py) - REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for $\pi_*$ 
- [REINFORCE_baseline](https://github.com/romanoneg/JaxRL/blob/main/RLbook/REINFORCE_baseline.py) - REINFORCE with Baseline (episodic), for estimating $\pi_\theta \approx \pi_*$\
      * The gradient descent formulated in the book seemed to not work as well as pure mse, implemented both
- [Batched_Actor-Critic](https://github.com/romanoneg/JaxRL/blob/main/RLbook/B-Actor-Critic.py) - \[Batched\] Actor–Critic (episodic), for estimating $\pi_\theta \approx \pi_*$\
	* ran into some issues implementing the one from the book, opted for Batched actor-critic algorithm)\
	Training does seem, somehow, less stable than normal REINFORCE or REINFORCE with baseline.)\
	Likely due to the simplicity of the environment? Will have to do more testing. 

---

(image from [Firefly beta](https://www.adobe.com/sensei/generative-ai/firefly.html))
