import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import chex
from typing import Dict, List
import numpy as np
from flax import struct

# Define SHIFTS: action index to movement vector
SHIFTS = np.array([
    [-1, 0],  # up
    [1, 0],   # down
    [0, -1],  # left
    [0, 1],   # right
])

# Define the State dataclass
@struct.dataclass
class State:
    agent_pos: Dict[str, chex.Array]  # {'agent_name': np.array([x, y])}
    goal_pos: chex.Array              # shape (g, 2)
    goal_taken: chex.Array            # shape (g,)
    t: int = 0

def get_triangle_coords(x, y, action_idx, cell_size):
    cx, cy = x + 0.5, y + 0.5
    s = 0.4 * cell_size

    if action_idx not in [0, 1, 2, 3]:
        # fallback to random orientation if invalid action
        angle = np.random.uniform(0, 2 * np.pi)
        return [(cx + s * np.cos(angle),
                 cy + s * np.sin(angle)),
                (cx + s * np.cos(angle + 2 * np.pi / 3),
                 cy + s * np.sin(angle + 2 * np.pi / 3)),
                (cx + s * np.cos(angle + 4 * np.pi / 3),
                 cy + s * np.sin(angle + 4 * np.pi / 3))]

    dx, dy = SHIFTS[action_idx]

    if dx == -1 and dy == 0:  # UP
        return [(cx, cy + s), (cx - s, cy - s), (cx + s, cy - s)]
    elif dx == 1 and dy == 0:  # DOWN
        return [(cx, cy - s), (cx - s, cy + s), (cx + s, cy + s)]
    elif dx == 0 and dy == -1:  # LEFT
        return [(cx - s, cy), (cx + s, cy + s), (cx + s, cy - s)]
    elif dx == 0 and dy == 1:  # RIGHT
        return [(cx + s, cy), (cx - s, cy + s), (cx - s, cy - s)]

def draw_frame(ax, state: State, actions: Dict[str, int], grid_size):
    ax.clear()
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw grid background
    for x in range(grid_size[0]):
        for y in range(grid_size[1]):
            ax.add_patch(patches.Rectangle((x, y), 1, 1, linewidth=0.3,
                                           edgecolor='gray', facecolor='black'))

    # Draw goals
    for i, pos in enumerate(state.goal_pos):
        if state.goal_taken[i] == 0:
            ax.add_patch(patches.Rectangle((pos[0], pos[1]), 1, 1,
                                           facecolor='#39FF14', edgecolor='gray', linewidth=0.3))

    # Draw agents
    for agent_name, pos in state.agent_pos.items():
        x, y = pos
        action = actions.get(agent_name, -1)
        coords = get_triangle_coords(x, y, action, 1)
        is_adversary = (agent_name == 'adversary')
        color = 'red' if is_adversary else 'blue'
        triangle = patches.Polygon(coords, closed=True, color=color)
        ax.add_patch(triangle)

def create_animation(states: List[State], actions_list: List[Dict[str, int]], grid_size):
    fig, ax = plt.subplots(figsize=(grid_size[0], grid_size[1]))

    def update(frame_idx):
        draw_frame(ax, states[frame_idx], actions_list[frame_idx], grid_size)

    anim = animation.FuncAnimation(
        fig, update, frames=len(states), interval=500, repeat=False)
    return anim


# Example usage:
# Assume you have a list of State objects `states`
# and a list of corresponding actions dictionaries `actions_list`
# and your grid is 5x5:

if __name__ == "__main__":
    # Dummy data to visualize
    states = [
        State(
            agent_pos={"agent1": np.array([0, 0]),
                    "agent2": np.array([2, 0]),
                    "adversary": np.array([1, 2])},
            goal_pos=np.array([[0, 2], [2, 2]]),
            goal_taken=np.array([0, 0]),
            t=0
        ),
        State(
            agent_pos={"agent1": np.array([0, 0]),
                    "agent2": np.array([1, 0]),
                    "adversary": np.array([0, 2])},
            goal_pos=np.array([[0, 2], [2, 2]]),
            goal_taken=np.array([1, 0]),
            t=1
        ),
        State(
            agent_pos={"agent1": np.array([0, 1]),
                    "agent2": np.array([1, 1]),
                    "adversary": np.array([0, 1])},
            goal_pos=np.array([[0, 2], [2, 2]]),
            goal_taken=np.array([1, 0]),
            t=2
        ),
    ]

    actions_list = [
        {"agent1": 0, "agent2": 0, "adversary": 0},  # t=0 → t=1
        {"agent1": 3, "agent2": 3, "adversary": 2},  # t=1 → t=2
        {"agent1": 3, "agent2": 3, "adversary": 1},  # t=2 → t=3
    ]


    anim = create_animation(states, actions_list, grid_size=(8, 8))
    plt.show()
