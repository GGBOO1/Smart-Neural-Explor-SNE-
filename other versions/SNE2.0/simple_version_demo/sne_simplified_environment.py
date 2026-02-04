import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SimplifiedEnvironment:
    def __init__(self, state_dim=4, action_dim=2, max_steps=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_steps = max_steps
        self.reset()
        self.fig = None
        self.ax = None
        self.line = None
        self.steps = []
    def reset(self):
        self.state = np.random.uniform(-1, 1, self.state_dim)
        self.steps_taken = 0
        self.steps = [self.state.copy()]
        return self.state
    def step(self, action):
        self.steps_taken += 1
        reward = -np.linalg.norm(self.state[:len(action)] - action)
        state_update = np.zeros(self.state_dim)
        state_update[:len(action)] = action
        self.state = self.state * 0.9 + state_update * 0.1
        self.state = np.clip(self.state, -1, 1)
        done = self.steps_taken >= self.max_steps
        self.steps.append(self.state.copy())
        return self.state, reward, done
    def render(self, mode='human'):
        if mode == 'human':
            print(f"State: {self.state}, Steps: {self.steps_taken}")
        elif mode == 'plot':
            if not self.fig:
                self.fig, self.ax = plt.subplots()
                self.ax.set_xlim(-1.1, 1.1)
                self.ax.set_ylim(-1.1, 1.1)
                self.ax.set_xlabel('Dimension 1')
                self.ax.set_ylabel('Dimension 2')
                self.ax.set_title('SNE Agent Trajectory')
                self.line, = self.ax.plot([], [], 'b-o')
            steps = np.array(self.steps)
            if len(steps) > 1:
                self.line.set_data(steps[:, 0], steps[:, 1])
                self.fig.canvas.draw()
                plt.pause(0.01)
    def animate(self, filename='sne_trajectory.mp4'):
        if len(self.steps) < 2:
            print("Not enough steps to animate")
            return
        fig, ax = plt.subplots()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('SNE Agent Trajectory')
        line, = ax.plot([], [], 'b-o')
        steps = np.array(self.steps)
        def update(frame):
            line.set_data(steps[:frame+1, 0], steps[:frame+1, 1])
            return line,
        anim = FuncAnimation(fig, update, frames=len(steps), interval=100, blit=True)
        anim.save(filename, writer='ffmpeg', fps=10)
        plt.close(fig)