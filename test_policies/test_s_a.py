import bisect

import numpy as np
from EthicalGatheringGame import MAEGG, NormalizeReward
from EthicalGatheringGame.presets import tiny, small, medium, large
from IndependentPPO import IPPO
from IndependentPPO.config import args_from_json
from IndependentPPO.agent import SoftmaxActor
from IndependentPPO.ActionSelection import *
import torch as th
import matplotlib
import os

matplotlib.use('TkAgg')
import gym

eff_rate = 1
db = 1
we = 10
large["n_agents"] = 5
large["donation_capacity"] = db
large["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate * 5)
large["we"] = [1, we]
large["color_by_efficiency"] = True

# If root dir is not MAEGG_IPPO, up one level
current_directory = os.getcwd()
directory_name = os.path.basename(current_directory)
while directory_name != "MAEGG_IPPO":
    os.chdir("..")
    current_directory = os.getcwd()
    directory_name = os.path.basename(current_directory)
print(current_directory)

# Loading the agents
path = f"ECAI/db{db}_effrate{eff_rate}_we{we}_ECAI/db{db}_effrate{eff_rate}_we{we}_ECAI/2500_100000_1_(1)"
args = args_from_json(path + "/config.json")
agents = IPPO.actors_from_file(path)


def normalize_obs(obs, db, ap):
    obs_shape = np.prod(obs.shape)
    obs = np.reshape(obs, (obs_shape,))
    normalized_obs = np.ones(obs_shape)
    normalized_obs[obs == '@'] = 0.5
    # Set middle to 0.75
    normalized_obs[obs.shape[0] // 2] = 0.75
    normalized_obs[obs == ' '] = 0.25
    normalized_obs[obs == '='] = 0

    donation_box_states = list(range(args.n_agents + 1)) + [large["donation_capacity"]]
    normalized_db_state = bisect.bisect_right(donation_box_states, db) - 1
    normalized_db_state /= (len(donation_box_states) - 1)

    # Normalize donation box and survival status
    aux = np.zeros(2)
    aux[0] = normalized_db_state
    if ap == 0:
        aux[1] = 0
    elif ap < large["survival_threshold"]:
        aux[1] = 0.25
    elif ap == large["survival_threshold"]:
        aux[1] = 0.75
    else:
        aux[1] = 1

    normalized_obs = np.concatenate((normalized_obs, aux))
    return normalized_obs


"""
# Get agents probabilities
x = th.tensor(sample, dtype=th.float32)
with th.no_grad():
    prob = agents[0].forward(x)
    
    
if ag.apples == 0:
aux[1] = 0
elif ag.apples < self.survival_threshold:
aux[1] = 0.25
elif ag.apples == self.survival_threshold:
aux[1] = 0.75
else:
aux[1] = 1

donation_box_states = list(range(self.n_agents + 1)) + [self.donation_capacity]
normalized_db_state = bisect.bisect_right(donation_box_states, self.donation_box) - 1
normalized_db_state /= (len(donation_box_states) - 1)
"""

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PixelGridApp:
    def __init__(self, root):
        self.root = root
        self.root.title("9x9 Pixel Grid")

        self.grid_size = 9
        self.cells = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.colors = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.color_cycle = ['green', 'red', 'black']
        self.color_mapping = {'green': '@', 'red': 'A', 'black': ' '}

        self.create_grid_frame()
        self.create_controls_frame()
        self.create_plot_frame()

    def create_grid_frame(self):
        self.grid_frame = tk.Frame(self.root)
        self.grid_frame.pack(side=tk.LEFT, padx=10, pady=10)

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = tk.Label(self.grid_frame, bg="black", width=4, height=2, borderwidth=1, relief="solid",
                                highlightbackground="white", highlightcolor="white", highlightthickness=1)
                cell.grid(row=row, column=col)
                cell.bind("<Button-1>", lambda event, r=row, c=col: self.change_color(event, r, c))
                self.cells[row][col] = cell
                self.colors[row][col] = ' '  # Initialize with blank space for black

    def change_color(self, event, row, col):
        current_color = self.cells[row][col].cget("bg")
        next_color = self.color_cycle[(self.color_cycle.index(current_color) + 1) % len(self.color_cycle)]
        self.cells[row][col].config(bg=next_color)
        self.colors[row][col] = self.color_mapping[next_color]

    def create_controls_frame(self):
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.input1_label = tk.Label(self.controls_frame, text="Input 1:")
        self.input1_label.grid(row=0, column=0, sticky=tk.E)

        self.input1 = tk.Entry(self.controls_frame)
        self.input1.grid(row=0, column=1)

        self.input2_label = tk.Label(self.controls_frame, text="Input 2:")
        self.input2_label.grid(row=1, column=0, sticky=tk.E)

        self.input2 = tk.Entry(self.controls_frame)
        self.input2.grid(row=1, column=1)

        explanation = (
            "Grid Game Rules:\n"
            "1. Click on any cell to change its color.\n"
            "2. The colors cycle through: green -> red -> black.\n"
            "3. All cells start as black.\n"
            "4. Use the 'Execute Function' button to process the grid.\n"
        )
        text_label = tk.Label(self.controls_frame, text=explanation, justify=tk.LEFT, padx=10, pady=10)
        text_label.grid(row=2, column=0, columnspan=2, sticky=tk.W)

        button = tk.Button(self.controls_frame, text="Execute Function", command=self.execute_function)
        button.grid(row=3, column=0, columnspan=2, pady=10)

    def create_plot_frame(self):
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure, self.ax = plt.subplots(figsize=(15, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot(self):
        # Placeholder data: 5 distributions with 7 options
        data = np.random.rand(5, 7)
        data = data / data.sum(axis=1, keepdims=True)  # Normalize to create probability distributions

        self.ax.clear()
        x = np.arange(7)  # 7 discrete options
        bar_width = 0.15
        colors = ['b', 'g', 'r', 'c', 'm']

        for i in range(5):
            self.ax.bar(x + i * bar_width, data[i], bar_width, label=f'Distribution {i + 1}', color=colors[i])

        self.ax.set_xlabel('Options')
        self.ax.set_ylabel('Probability')
        self.ax.set_title('Probability Distributions')
        self.ax.set_xticks(x + bar_width * 2)
        self.ax.set_xticklabels([f'Option {i + 1}' for i in range(7)])
        self.ax.legend()
        self.canvas.draw()

    def execute_function(self):
        input1_value = int(self.input1.get()) if self.input1.get().isdigit() else None
        input2_value = int(self.input2.get()) if self.input2.get().isdigit() else None

        # For demonstration, print the internal representation of the grid and the inputs
        print(f"Input 1: {input1_value}, Input 2: {input2_value}")
        for row in self.colors:
            print(row)

        # Normalize the grid and inputs
        x = normalize_obs(np.array(self.colors), input1_value, input2_value)
        print(x)
        self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = PixelGridApp(root)
    root.mainloop()
