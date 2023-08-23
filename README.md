# JaxRL - romanoneg

![Firefly cute robots learning to play soccer 81139](https://github.com/romanoneg/JaxRL/assets/43445765/6decdc4c-4b5c-467c-8240-bddba3506a16)


learning jax, specifically for RL

Very Very simple RL agents built in Jax, with (sometimes!) pytorch equivalents.

### -------------------------------First Tests, REINFORCE agents--------------------------------------

As a very quick preliminary test here are the times each script took to solve `CartPole-v1` :

| Script        | Time     | % change | Notes |
|--------------|-----------|------------|-----------|
| REINFORCE_pytorch      | 9m12s   | -- | by far the easiest to pickup        |
| REINFORCE_jax          | 58.6s  | ~90% faster | includes compile time       |
| vmap_REINFORCE_jax     | 🔥**40.0s**🔥 | ~93% faster | include compile time       |
| pmap_vmap_REINFORCE_jax| --            | Prob really fast| my second GPU is too old to test 🙃  |

(Jax Scripts use the [gymnax](https://github.com/RobertTLange/gymnax) library)
