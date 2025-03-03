import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os

class FrozenLakeEnv:
    def __init__(self, map_path):
        '''
        Environment class for FrozenLake

        Top Left Corner: (0, 0)
        Bottom Right Corner: (map_size - 1, map_size - 1)
        '''
        self.map_name = os.path.splitext(os.path.basename(map_path))[0]
        self.map = np.genfromtxt(map_path, dtype=str, delimiter='\n')
        self.map = np.array([list(row) for row in self.map])
        self.map_x, self.map_y = self.map.shape

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # left, right, up, down

        # Variables
        self.state = (x, y) = self.get_start_state()
        assert self.state is not None, 'Start state not found'

    def get_start_state(self):
        for x in range(self.map_x):
            for y in range(self.map_y):
                if self.map[y, x] == 'S':
                    return (x, y)
        return None
    
    def reset(self):
        self.state = self.get_start_state()
        return self.state
    
    def step(self, action):
        x, y = self.state
        dx, dy = self.actions[action]
        new_x, new_y = x + dx, y + dy

        reward = 0
        done = False

        reward -= 0 # step cost, can use 0.01
        if new_x < 0 or new_x >= self.map_x or new_y < 0 or new_y >= self.map_y: # out of bounds, dont move
            reward -= 0 # out of bounds cost, can use 0.01
            return (x, y), reward, done
        
        new_tile = self.map[new_y, new_x] # ., H, G, S -> ice, hole, goal, start
        if new_tile == 'H': # fell into hole
            self.state = (new_x, new_y)
            reward -= 1
            done = True
        elif new_tile == 'G': # reached goal
            self.state = (new_x, new_y)
            reward += 1
            done = True
        elif new_tile == '.' or new_tile == 'S': # move
            self.state = (new_x, new_y)
            reward -= 0
        else:
            raise Exception('Unknown cell type')
        
        return (new_x, new_y), reward, done
    
    def visualize_q_table(self, episode, Q_table, algo_type):
        """Visualizes the Q-table with each cell divided into four triangles and labeled."""
        map_size = self.map_x
        Q_font_size = 8 * (map_size**0.5 / 4**0.5)
        triangle_line_width = 0.5 * (map_size**0.5 / 4**0.5)
        tile_font_size = 12 * (map_size**0.5 / 4**0.5)
        save_path = f"frames/{algo_type}_{self.map_name}"
        os.makedirs(save_path, exist_ok=True)

        fig, ax = plt.subplots(figsize=(map_size*2, map_size*2))
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_xticks(range(map_size+1))
        ax.set_yticks(range(map_size+1))
        ax.grid(True, color='black', linewidth=1.5)
        
        max_q = max([max(q_values) for q_values in Q_table.values()])
        min_q = min([min(q_values) for q_values in Q_table.values()])
        
        for (x, y), q_values in Q_table.items():
            q_values = np.array(q_values)
            norm_q = (q_values - min_q) / (max_q - min_q + 1e-6)  # Normalize
            colors = [(0.2 * norm, 0.8 * norm, 0.2 * norm) for norm in norm_q]  # Richer green     

            # Flip y-axis to match (0,0) at top left
            y = map_size - 1 - y
            
            triangles = [
                [(x, y), (x+0.5, y+0.5), (x, y+1)],   # Left (0)
                [(x+1, y), (x+0.5, y+0.5), (x+1, y+1)], # Right (1)
                [(x, y+1), (x+0.5, y+0.5), (x+1, y+1)], # Up (2)
                [(x, y), (x+0.5, y+0.5), (x+1, y)]      # Down (3)
            ]
                        
            for i, triangle in enumerate(triangles):
                patch = patches.Polygon(triangle, closed=True, facecolor=colors[i], edgecolor='green', linewidth=triangle_line_width)
                ax.add_patch(patch)
                
                # Add action-value label inside triangle
                ax.text(np.mean([p[0] for p in triangle]), np.mean([p[1] for p in triangle]), 
                        f"{Q_table[(x, map_size - 1 - y)][i]:.2f}",
                        ha='center', va='center', fontsize=Q_font_size, color='white')
            
            # Add cell type label only if it is S, H, or G
            cell_type = self.map[self.map_y - 1 - y, x]
            if cell_type in ['S', 'H', 'G']:
                ax.text(x+0.5, y+0.5, cell_type, ha='center', va='center',
                        fontsize=tile_font_size, fontweight='bold', color='black')
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Episode {episode}")
        plt.savefig(f"{save_path}/frame_{episode:04d}.png")
        plt.close()

    def generate_gif(self, algo_type):
        """Generates a GIF from saved frames and keeps only the last frame."""
        save_path = f"frames/{algo_type}_{self.map_name}"
        gif_name = f"frames/{algo_type}.gif"
        images = sorted(os.listdir(save_path))
        frames = [imageio.imread(os.path.join(save_path, img)) for img in images]
        imageio.mimsave(gif_name, frames, duration=0.1)
        
        # Keep only the last frame
        for img in images[:-1]:
            os.remove(os.path.join(save_path, img))

    def visualize_deterministic_policy(self, policy_table, algo_type):
        """Visualizes the deterministic policy with arrows and colored cells."""
        map_size = self.map_x
        save_path = f"frames/{algo_type}_{self.map_name}"
        os.makedirs(save_path, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(map_size * 2, map_size * 2))
        ax.set_xlim(0, map_size)
        ax.set_ylim(0, map_size)
        ax.set_xticks(range(map_size + 1))
        ax.set_yticks(range(map_size + 1))
        ax.grid(True, color='black', linewidth=1.5)

        # Define colors
        cell_colors = {'.': 'white', 'H': 'black', 'G': 'green', 'S': 'white'}
        text_colors = {'H': 'white', 'G': 'white', '.': 'black', 'S': 'black'}

        # Action arrows
        action_arrows = {0: "⬅️", 1: "➡️", 2: "⬆️", 3: "⬇️"}

        for x in range(map_size):
            for y in range(map_size):
                true_y = map_size - 1 - y  # Flip y-axis
                
                cell_type = self.map[y, x]
                color = cell_colors[cell_type]
                text_color = text_colors[cell_type]

                # Draw colored cell
                ax.add_patch(patches.Rectangle((x, true_y), 1, 1, color=color, ec='black', lw=1.5))

                # Place the cell label or action
                if cell_type in ['S', 'H', 'G']:
                    ax.text(x + 0.5, true_y + 0.5, cell_type, ha='center', va='center',
                            fontsize=24, fontweight='bold', color=text_color)
                else:
                    state = (x, y)
                    if state in policy_table:
                        action = int(np.argmax(policy_table[state]))  # Greedy action
                        ax.text(x + 0.5, true_y + 0.5, action_arrows[action], 
                                ha='center', va='center', fontsize=24, color='black')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Deterministic Policy - {algo_type}")
        plt.savefig(f"{save_path}/deterministic_policy.png")
        plt.close()

