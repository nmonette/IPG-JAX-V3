import jaxmarl
import jax
import jax.numpy as jnp
import numpy as np
import chex
from typing import Tuple, Optional, Dict
from functools import partial

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import MultiDiscrete, Discrete
from flax import struct

#N DEFAULT_PARAMS
NUM_TEAMMATES: int = 2
NUM_GOALS: int = 2
GRID_SIZE: int = 5
MAX_STEPS: int = 25

SHIFTS = jnp.array([
    [-1, 0], # up
    [1, 0],  # down
    [0, -1], # left
    [0, 1],  # right
])

@struct.dataclass
class State:
    agent_pos: Dict[str, chex.Array] # agent to position mapping
    goal_pos: chex.Array  # position of the goals, shape (num_goals, 2)
    goal_taken: chex.Array # array of shape (num_goals, ) -> 0 if not taken, 1 if taken by team, -1 if by adversary
    t: int = 0

class AdvGridEnv(MultiAgentEnv):

    def __init__(
        self,  
        num_teammates: int = NUM_TEAMMATES,
        num_goals: int = NUM_GOALS,
        grid_size: int = GRID_SIZE,
        max_episode_steps: int = MAX_STEPS,
        **kwargs
        # collision_handling: str = "probabilistic" 
    ):
        self.num_teammates = num_teammates
        self.num_agents = num_teammates + 1
        self.num_goals = num_goals
        self.max_episode_steps = max_episode_steps

        self.num_entities = self.num_agents + num_goals
        self.agent_range = jnp.arange(self.num_agents, dtype=jnp.int32)

        self.agents = [f"agent_{i}" for i in range(self.num_teammates)] + ["adversary_0"]

        # observation is the position of the agent and the goals, as well as if the goals have been terminated
        # so: (GOAL_1_x, GOAL_1_y, GOAL_1_TAKEN, ..., AGENT_x, AGENT_y)
        agent_obs = (grid_size, grid_size)
        goal_obs = (grid_size, grid_size, 2) * num_goals
        full_obs = goal_obs + agent_obs
        self.observation_spaces = {
            agent: MultiDiscrete(full_obs) for agent in self.agents
        }
        self.action_spaces = {
            agent: Discrete(4) for agent in self.agents
        }

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, rng: chex.PRNGKey, state: State, actions: dict):

        rng, _rng = jax.random.split(rng)
        state = self._apply_actions(_rng, state, actions)

        reward = self.rewards(state)
        obs = self.get_obs(state)
        info = {}
        done = jnp.logical_or(reward["adversary_0"] != 0, state.t >= self.max_episode_steps)
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        state = state.replace(
            t = state.t + 1
        )

        return obs, state, reward, dones, info

    @partial(jax.jit, static_argnums=[0])
    def reset(self, rng: chex.PRNGKey) -> Tuple[chex.Array, State]:

        rng, _rng = jax.random.split(rng)

        def _place_loop(carry, _):
            grid, rng = carry

            a = jnp.arange(GRID_SIZE)
            open_rows = jnp.logical_not((grid == 0).all(axis=1))

            rng, _rng = jax.random.split(rng)
            x = jax.random.choice(
                _rng, a, replace=False, p = open_rows / open_rows.sum()
            )

            rng, _rng = jax.random.split(rng)
            y = jax.random.choice(
                _rng, a, replace=False, p = grid[x] / grid[x].sum()
            )

            grid = grid.at[x, y].set(0)

            return (grid, rng), jnp.array([x, y])
        
        # Initialize the grid
        grid = jnp.ones((GRID_SIZE, GRID_SIZE), dtype=jnp.int32)
        rng, _rng = jax.random.split(rng)
        positions = jax.lax.scan(_place_loop, (grid, _rng), xs=None, length=self.num_entities)[1]

        state = State(
            agent_pos={agent: positions[-i-1] for i, agent in enumerate(self.agents)},
            goal_pos = positions[:-self.num_agents],
            goal_taken = jnp.zeros(self.num_goals, dtype=jnp.int32)
        )

        return self.get_obs(state), state
    
    def get_obs(self, state: State) -> Dict[str, chex.Array]:

        goal_obs = jnp.concatenate([state.goal_pos, state.goal_taken[:, None]], axis=1).reshape(-1)

        agent_positions = jnp.stack([state.agent_pos[agent] for agent in self.agents])

        @jax.vmap
        def _observation(agent_pos):
            return jnp.concatenate([goal_obs, agent_pos])

        obs = _observation(agent_positions)
        return {a: obs[i] for i, a in enumerate(self.agents)}


    def _apply_actions(self, rng: chex.PRNGKey, state: State, actions: dict) -> State:

        actions = jnp.array([actions[agent] for agent in self.agents])
        positions = jnp.stack([state.agent_pos[agent] for agent in self.agents])
        # Move the agents (random ordering as a tiebreaker)
        def _move_loop(pos, agent_data):

            action, prev_pos, agent_idx = agent_data
            new_pos = prev_pos + SHIFTS[action.squeeze()]

            # Ensure the new position is within bounds
            new_pos = jnp.clip(new_pos, 0, GRID_SIZE - 1)
            
            # check for collisions
            collision = (new_pos == positions).all(axis=1).any()
            return pos.at[agent_idx].set(jax.lax.select(collision, prev_pos, new_pos)), None
        
        rng, _rng = jax.random.split(rng)
        perm = jax.random.permutation(_rng, self.agent_range)
        agent_data = jax.tree_util.tree_map(lambda x: x[perm], (actions, positions, self.agent_range))
        agent_pos = jax.lax.scan(_move_loop, positions, agent_data)[0]        
        # Check goals
        taken_fn = lambda pos, goals, is_adversary: (
            jax.lax.select(
                jnp.logical_and((pos == goals).all(axis=1), jnp.logical_not(state.goal_taken)), 1, 0
            ) * jax.lax.select(is_adversary, -1, 1) 
        )
        new_taken = jax.vmap(taken_fn, in_axes=(0, None, 0))(agent_pos, state.goal_pos, jnp.zeros(self.num_agents, dtype=bool).at[-1].set(1)).sum(axis=0)
        goal_taken = jnp.where(state.goal_taken != 0, state.goal_taken, new_taken)
        agent_pos = {agent: agent_pos[i] for i, agent in enumerate(self.agents)}

        state = State(
            agent_pos=agent_pos,
            goal_pos=state.goal_pos,
            goal_taken=goal_taken
        )

        return state
    
    def rewards(self, state: State) -> Dict[str, float]:
        """Assign rewards for all agents"""

        adv_reward = jax.lax.select((state.goal_taken == -1).any(), 1, 0)
        team_reward = jax.lax.select((state.goal_taken == 1).all(), 1, -adv_reward)

        r = jnp.full(self.num_agents, team_reward).at[-1].set(-team_reward)
        return {agent: r[i] for i, agent in enumerate(self.agents)}








            
            



