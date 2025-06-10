import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import struct
from flax.training.train_state import TrainState
import wandb
import hydra
from omegaconf import OmegaConf

from functools import partial
from math import prod

import distrax
import jaxmarl
from environments.adv_grid import AdvGridEnv


def projection_simplex_truncated(x: jnp.ndarray, eps: float) -> jnp.ndarray: 
    """
    Code adapted from 
    https://www.ryanhmckenna.com/2019/10/projecting-onto-probability-simplex.html
    To represent truncated simplex projection. Assumes 1D vector. 
    """
    ones = jnp.ones_like(x)
    lambdas = jnp.concatenate((ones * eps - x, ones - x), axis=-1)
    idx = jnp.argsort(lambdas)
    lambdas = jnp.take_along_axis(lambdas, idx, -1)
    active = jnp.cumsum((jnp.float32(idx < x.shape[-1])) * 2 - 1, axis=-1)[..., :-1]
    diffs = jnp.diff(lambdas, n=1, axis=-1)
    left = (ones * eps).sum(axis=-1)
    left = left.reshape(*left.shape, 1)
    totals = left + jnp.cumsum(active*diffs, axis=-1)

    def generate_vmap(counter, func):
        if counter == 0:
            return func
        else:
            return generate_vmap(counter - 1, jax.vmap(func))
                
    i = jnp.expand_dims(generate_vmap(len(totals.shape) - 1, partial(jnp.searchsorted, v=1))(totals), -1)
    lam = (1 - jnp.take_along_axis(totals, i, -1)) / jnp.take_along_axis(active, i, -1) + jnp.take_along_axis(lambdas, i+1, -1)
    return jnp.clip(x + lam, eps, 1)

class ProjectionTrainState(TrainState):
    trunc_size: float = 1e-5

    def apply_gradients(self, grads, **kwargs):
        """
        Apply gradients with projection onto simplex.
        """
        ts = super().apply_gradients(grads = jax.tree_util.tree_map(jnp.zeros_like, self.params))
        m = 1
        if kwargs.get("team"):
            m = 10
        new_params = jax.tree_util.tree_map(
            lambda x, g: projection_simplex_truncated(x + m * g, self.trunc_size), ts.params, grads
        )
        return ts.replace(
            params=new_params,
        )

class TabularPolicy(nn.Module):
    state_space: tuple[int]
    num_states: int
    num_actions: int

    def setup(self):
        self.params = self.param('params', lambda _, shape: jnp.full(shape, 1 / self.num_actions), (self.num_states, self.num_actions))

    def __call__(self, state):

        def fn(state):
            idx = jnp.ravel_multi_index(state, self.state_space, mode="clip")
            return idx
        probs = self.params[jax.vmap(fn)(state.reshape(-1, state.shape[-1]))]

        pi = distrax.Categorical(probs=probs)

        return pi    

class Policy(nn.Module):
    state_space: tuple[int]
    num_states: int
    num_actions: int

    @nn.compact
    def __call__(self, state):
        
        policies = nn.vmap(
            TabularPolicy,
            in_axes=0, out_axes=0,
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(self.state_space, self.num_states, self.num_actions)

        return policies(state)
    
@struct.dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

# def batchify(x: dict, agent_list, num_actors):
#     max_dim = max([x[a].shape[-1] for a in agent_list])
#     def pad(z, length):
#         return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

#     x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
#     # return x.reshape((num_actors, -1))
#     return x

def batchify(x: dict, agent_list, *args, **kwargs):
    return jnp.stack([x[a] for a in agent_list], axis=0)

def unbatchify(x: jnp.ndarray, agent_list, num_envs, *args, **kwargs):
    x = x.reshape((len(agent_list), num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}
    
def make_train(config):

    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = AdvGridEnv(**config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    ADV_IDX = env.agents.index("adversary_0")
    NUM_STATES = prod(env.observation_space(env.agents[0]).shape)
    
    def linear_schedule(count):
        frac = 1.0 - count / config["NUM_UPDATES"]
        return config["LR"] * frac
    
    def train(rng):

        # INIT NETWORK
        obs_space = tuple(env.observation_space(env.agents[0]).num_categories)
        network = Policy(obs_space, prod(env.observation_space(env.agents[0]).num_categories), env.action_space(env.agents[0]).n)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((env.num_agents, 1, env.observation_space(env.agents[0]).shape[-1]), dtype=jnp.int32)
        # init_x = jax.ShapeDtypeStruct((env.num_agents, env.observation_space(env.agents[0])), dtype=jnp.int32)
        network_params = network.init(_rng, init_x)

        train_state = ProjectionTrainState.create(
            apply_fn=network.apply,
            params=network_params,
            # tx=optax.scale_by_schedule(linear_schedule),
            tx = optax.scale(config["LR"])
        )

        def compute_returns(rewards, dones):

            def loop(returns, data):
                reward, done = data

                returns = reward + returns * config["GAMMA"] * (1 - done)
                return returns, returns

            ret = jax.lax.scan(loop, jnp.zeros_like(rewards[0]), (rewards, dones), reverse=True)[1]
            return ret
        
        # INIT ENV
        def rollout_fn(rng, train_state, collect_lambda=False):

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, lambda_, gammas = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi = network.apply(train_state.params, obs_batch)
                value = None

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                if collect_lambda:
                    def update_lambda(lam, obs, action, gamma):
                        idx = jnp.ravel_multi_index(obs, obs_space, mode="clip")
                        lam = lam.at[idx, action].add(gamma)
                        return lam
                    
                    lambda_ = jax.vmap(update_lambda, in_axes=(None, 0, 0, 0))(lambda_, last_obs["adversary_0"], env_act["adversary_0"], gammas).mean(0)
                    

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )

                gammas = jnp.where(1 - done["adversary_0"], gammas * config["GAMMA"], jnp.ones_like(gammas))
                runner_state = (train_state, env_state, obsv, rng, lambda_, gammas)
                return runner_state, transition
            
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset)(reset_rng)

            lambda_ = jnp.zeros((NUM_STATES, env.action_space(env.agents[ADV_IDX]).n))
            runner_state = (train_state, env_state, obsv, rng, lambda_, jnp.ones((config["NUM_ENVS"],)))

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            if collect_lambda:
                lambda_ = runner_state[4]
                return traj_batch, lambda_
            
            return traj_batch
        
        def adv_update(rng, train_state, ADV_IDX=ADV_IDX):

            orig_traj_batch, lambda_ = rollout_fn(rng, train_state, collect_lambda=True)

            # separate out adversarial index
            traj_batch = jax.tree.map(lambda x: x[:, ADV_IDX] if len(x) > 0 else x, orig_traj_batch)

            def regularize_rewards(rewards, obvs, actions, lambda_):
                
                def reg(reward, obs, action):

                    idx = jnp.ravel_multi_index(obs, obs_space, mode="clip")
                    lam = lambda_[idx, action]
                    return reward - config["NU"] * lam
                
                flat_rewards = rewards.flatten()
                return jax.vmap(reg)(flat_rewards, obvs.reshape(len(flat_rewards), -1), actions.flatten()).reshape(*rewards.shape)
            
            rewards = regularize_rewards(traj_batch.reward, traj_batch.obs, traj_batch.action, lambda_)

            # COMPUTE RETURNS
            targets = compute_returns(rewards, dones=traj_batch.done)

            def loss_fn(params):

                pi = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(params, orig_traj_batch.obs)
                log_probs = pi.log_prob(orig_traj_batch.action)[:, ADV_IDX]
                policy_loss = (log_probs * targets).sum(0).mean()

                return policy_loss, (policy_loss, pi.entropy()[:, ADV_IDX].mean())  

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(train_state.params)
            old_params = train_state.params
            train_state = train_state.apply_gradients(grads=grads, team=True)

            loss_info = {
                "loss": total_loss[0],
                "entropy": total_loss[1][1],
                "adv_optimality": optax.tree_utils.tree_sum(jax.tree_util.tree_map(
                    lambda old, new: jnp.linalg.norm(old - new), old_params, train_state.params
                ))
            }

            return train_state, loss_info
        

        def team_update(rng, train_state):
            
            # COLLECT TRAJECTORIES
            traj_batch = rollout_fn(rng, train_state)

            # COMPUTE RETURNS
            targets = jax.vmap(compute_returns, in_axes=1, out_axes=1)(traj_batch.reward, traj_batch.done)

            def loss_fn(params):

                pi = jax.vmap(train_state.apply_fn, in_axes=(None, 0))(params, traj_batch.obs)
                log_probs = pi.log_prob(traj_batch.action).at[:, ADV_IDX].multiply(0)
                policy_loss = (log_probs * targets).sum(0).mean()


                mask = jnp.ones_like(pi.entropy()).at[:, ADV_IDX].set(0)
                return policy_loss, (policy_loss, pi.entropy().mean(where=mask))  

            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            total_loss, grads = grad_fn(train_state.params)
            old_params = train_state.params
            train_state = train_state.apply_gradients(grads=grads, team=True)

            loss_info = {
                "loss": total_loss[0],
                "entropy": total_loss[1][1],
                "team_optimality": optax.tree_utils.tree_sum(jax.tree_util.tree_map(
                    lambda old, new: jnp.linalg.norm(old - new), old_params, train_state.params
                ))
            }

            return train_state, loss_info
        
        def update_fn(rng, train_state):

            def adv_loop(carry, _):
                rng, train_state = carry
                rng, _rng = jax.random.split(rng)
                train_state, loss_info = adv_update(_rng, train_state)
                return (rng, train_state), loss_info
            
            rng, _rng = jax.random.split(rng)
            (_, train_state), adv_loss_info = jax.lax.scan(adv_loop, (_rng, train_state), None, config["NUM_ADV_STEPS"])

            rng, _rng = jax.random.split(rng)
            train_state, team_loss_info = team_update(_rng, train_state)

            loss_info = {
                "adv_loss": adv_loss_info["loss"][-1],
                "adv_entropy": adv_loss_info["entropy"][-1],
                "adv_optimality": adv_loss_info["adv_optimality"][-1],
                "team_loss": team_loss_info["loss"],
                "team_entropy": team_loss_info["entropy"],
                "team_optimality": team_loss_info["team_optimality"],
            }

            return (rng, train_state), loss_info
        
        def nash_gap(rng, train_state):

            rng, _rng = jax.random.split(rng)
            traj_batch = rollout_fn(_rng, train_state)

            base_performance = jax.vmap(compute_returns, in_axes=1, out_axes=1)(traj_batch.reward, traj_batch.done)[0].mean(axis=-1)

            def br_fn(rng, idx):
                
                def adv_loop(carry, _):
                    rng, train_state = carry
                    rng, _rng = jax.random.split(rng)
                    train_state, _ = adv_update(_rng, train_state, ADV_IDX=idx)
                    return (rng, train_state), None
            
                rng, _rng = jax.random.split(rng)
                (_, br_train_state), _ = jax.lax.scan(adv_loop, (_rng, train_state), None, config["NUM_ADV_STEPS"])

                br_traj_batch = jax.tree_util.tree_map(lambda x: x[:, idx], rollout_fn(rng, br_train_state))
                br_performance = compute_returns(br_traj_batch.reward, dones=br_traj_batch.done)

                return br_performance[0].mean(axis=-1)
            
            rng, _rng = jax.random.split(rng)
            br_performances = jax.vmap(br_fn, in_axes=(0, 0))(jax.random.split(_rng, env.num_agents), jnp.arange(env.num_agents))

            return (br_performances - base_performance).max()
        
        update = jax.jit(update_fn)
        gap_fn = jax.jit(nash_gap)

        best_iterate = train_state
        best_iterate_norm = jnp.inf

        for step in range(config["NUM_UPDATES"]):
            rng, _rng = jax.random.split(rng)
            (rng, train_state), loss_info = update(_rng, train_state)

            if loss_info["team_optimality"] < best_iterate_norm:
                best_iterate = train_state
                best_iterate_norm = loss_info["team_optimality"]

            if step % config["EVAL_FREQ"] == 0:
                rng, _rng = jax.random.split(rng)
                gap = gap_fn(_rng, train_state)
                loss_info["nash_gap"] = gap
                
                best_gap = gap_fn(_rng, best_iterate)
                loss_info["best_iterate_nash_gap"] = best_gap

                print(f"Step {step}: Nash Gap is {gap:.3f}, Best Iterate Nash Gap is {best_gap:.3f}")
                wandb.log(loss_info)

        return train_state

    return train
    
@hydra.main(version_base=None, config_path="config", config_name="ipg_tabular_grid")
def main(config):
    config = OmegaConf.to_container(config) 

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPGmax", "Tabular", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(50)
    train = make_train(config)
    out = train(rng)
    

if __name__ == "__main__":
    main()
        


