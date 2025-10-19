# %%
import gymnasium as gym
import numpy as np
import threading
import queue
import time
from tkinter import ttk
from collections import defaultdict
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from sympy import root

# %%
class QLearningBlackjackAgent:
    def __init__(self, env,
        learning_rate,
        initial_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor = 0.95):

        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.trainingError = []

    def getAction(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore action space
        else:
            return np.argmax(self.q_table[state])  # Exploit learned values
    
    def updateQVal(self, state, action, reward, terminated, nextState):
        futureQ = (not terminated) * np.max(self.q_table[nextState])

        target = reward + self.discount_factor * futureQ

        temporalDifference = target - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * temporalDifference

        self.trainingError.append(abs(temporalDifference))

    def decayEpsilon(self):
        # fix: update the correct attribute name
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# %%


# %%


def trainAgent(agent, progress_callback=None):
    """Train the agent. If progress_callback is provided it will be called as
    progress_callback(current_episode, total_episodes) from the training thread.
    Use a queue/from-mainloop polling to safely update the UI."""
    global episode_rewards; episode_rewards = []
    global episode_lengths; episode_lengths = []
    for episode in range(n_episodes):
        state, info = env.reset()
        terminated = False
        episode_reward = 0
        episode_length = 0

        while not terminated:
            action = agent.getAction(state)
            nextState, reward, terminated, truncated, info = env.step(action)
            agent.updateQVal(state, action, reward, terminated, nextState)
            state = nextState
            episode_reward += reward
            episode_length += 1

        agent.decayEpsilon()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # report progress to the UI (if provided) and yield the GIL so the
        # Tkinter mainloop can update the window while training proceeds.
        if progress_callback is not None:
            try:
                
                print(f"Training progress: Episode {episode + 1}/{n_episodes}")
                progress_callback(episode + 1, n_episodes)
            except Exception:
                # swallow UI update errors coming from background thread
                pass
        # Yield execution briefly to allow the main thread (Tkinter) to run.
        # This reduces UI starvation due to the GIL during tight CPU loops.
        time.sleep(0)


    # %%
from matplotlib import pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def getGraphs(agent):
    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        episode_rewards,
        rolling_length,
        "valid"
        )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        episode_lengths,
        rolling_length,
        "valid"
        )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    # `agent` must be provided so we can plot its training error history
    training_error_moving_average = get_moving_avgs(
        agent.trainingError,
        rolling_length,
        "same"
        )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()

# %%
def testAgent(agent, env, n_episodes=1000):
    totalRewards = []
    oldEpsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during testing

    for _ in range(n_episodes):
        state, info = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            action = agent.getAction(state)
            nextState, reward, terminated, truncated, info = env.step(action)
            state = nextState
            episode_reward += reward

        totalRewards.append(episode_reward)

    agent.epsilon = oldEpsilon  # Restore original epsilon
    win_rate = np.mean(np.array(totalRewards) > 0)
    average_reward = np.mean(totalRewards)
    return win_rate, average_reward, totalRewards

def showAgentRun(agent, delay_ms=500):
    """Show a visual run of the trained agent step-by-step using Tkinter.
    
    Uses after() so each step is visualized and GUI stays responsive.
    """
    oldEpsilon = agent.epsilon
    agent.epsilon = 0.0  # No exploration during testing
    runWindow = tk.Toplevel(root)
    runWindow.title("Trained Agent Run")
    label = tk.Label(runWindow)
    label.pack()
    
    state, info = env.reset()
    terminated = False
    cumulative_reward = 0
    
    def render_and_update():
        """Render frame and update label."""
        try:
            frame = env.render()
            if frame is not None:
                img = Image.fromarray(frame)
                tkimg = ImageTk.PhotoImage(img)
                label.config(image=tkimg)
                label.image = tkimg
            else:
                label.config(text=f"Player: {state[0]}, Dealer: {state[1]}, Reward: {cumulative_reward}")
        except Exception as e:
            label.config(text=f"Render error: {e}")
    
    def step():
        nonlocal state, terminated, cumulative_reward
        
        if terminated:
            print("Episode finished.")
            render_and_update()
            label.config(text=f"Episode finished. Final reward: {cumulative_reward}")
            agent.epsilon = oldEpsilon
            messagebox.showinfo("Run complete", f"Agent run finished. Final reward: {cumulative_reward}")
            runWindow.destroy()
            return
        
        # Take one step
        action = agent.getAction(state)
        nextState, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        state = nextState
        
        # Render and update immediately
        render_and_update()
        
        # Schedule next step
        runWindow.after(delay_ms, step)
    
    # Render initial state and start stepping
    render_and_update()
    runWindow.after(delay_ms, step)

 


def createTrainingWindow():
    for child in root.winfo_children():
        if isinstance(child, tk.Toplevel):
            child.destroy()
    global env; env = gym.make('Blackjack-v1', natural=naturalBlackjackVar.get(), sab=False, render_mode='rgb_array')
    # Create a Toplevel window so the main app (root) remains available
    secondWindow = tk.Toplevel(root)
    secondWindow.title("Training in Progress")
    label = tk.Label(secondWindow, text="Training the Q-Learning Agent...")
    label.pack()
    # Read parameters from the entry fields and update globals where needed
    try:
        lr = float(learning_rate_entry.get())
        episodes = int(float(n_episodes_entry.get()))
        start_eps = float(start_epsilon_entry.get())
        eps_decay = float(epsilon_decay_entry.get())
        final_eps = float(final_epsilon_entry.get())
    except Exception as e:
        messagebox.showerror("Invalid parameters", f"Error parsing hyperparameters: {e}")
        return

    # update global n_episodes used by trainAgent loop
    global n_episodes, learning_rate, start_epsilon, epsilon_decay, final_epsilon
    n_episodes = episodes
    learning_rate = lr
    start_epsilon = start_eps
    epsilon_decay = eps_decay
    final_epsilon = final_eps

    progress_q = queue.Queue()

    progress_label = tk.Label(secondWindow, text=f"0 / {n_episodes}")
    progress_label.pack(pady=(6, 0))

    progress_var = tk.DoubleVar(value=0)
    progressbar = ttk.Progressbar(secondWindow, maximum=n_episodes, variable=progress_var, length=400)
    progressbar.pack(padx=12, pady=6)
    # Make sure the progress window is shown before heavy work starts
    try:
        secondWindow.deiconify()
        secondWindow.lift()
        secondWindow.update_idletasks()
        secondWindow.update()
    except tk.TclError:
        pass
    # callback called from training thread to report progress
    def progress_callback(current, total):
        progress_q.put((current, total))

    def training_worker():

        agent = QLearningBlackjackAgent(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )
        # Run training (this is CPU-bound; run in a background thread)
        trainAgent(agent, progress_callback=progress_callback)
        # When finished, place the agent in the queue so UI can use it (e.g., plotting)
        progress_q.put(("done", agent))

    worker_thread = threading.Thread(target=training_worker, daemon=True)
    # start worker after forcing the window to render so the UI appears
    worker_thread.start()

    # Poll the queue and update UI
    def poll_queue():
        try:
            while True:
                item = progress_q.get_nowait()
                if isinstance(item, tuple) and item[0] == "done":
                    # training complete; item = ("done", agent)
                    _, trained_agent = item
                    progress_var.set(n_episodes)
                    progress_label.config(text=f"{n_episodes} / {n_episodes} - Done")
                    messagebox.showinfo("Training complete", "Training finished successfully.")
                    # Optionally show graphs
                    try:
                        secondWindow.destroy()
                        createEvaluationMenuWindow(trained_agent)
                    except Exception:
                        pass
                    return
                else:
                    current, total = item
                    progress_var.set(current)
                    progress_label.config(text=f"{current} / {total}")
        except queue.Empty:
            pass
        secondWindow.after(100, poll_queue)

    poll_queue()

def createEvaluationMenuWindow(trained_agent):
    evalMenu = tk.Toplevel(root)
    evalMenu.title("Evaluate Trained Agent")
    evalMenuLabel = tk.Label(evalMenu, text="Trained Agent using the following parameters:")
    evalMenuLabel.pack()
    evalMenuHyperparams = tk.Label(evalMenu, text=f"Learning Rate: {learning_rate}\n"
                                                  f"Number of Episodes: {n_episodes}\n"
                                                  f"Start Epsilon: {start_epsilon}\n"
                                                  f"Epsilon Decay: {epsilon_decay}\n"
                                                  f"Final Epsilon: {final_epsilon}\n"
                                                  f"Natural Blackjack: {naturalBlackjackVar.get()}")
    evalMenuHyperparams.pack()
    evalMenuShowRun = tk.Label(evalMenu, text="Show a run of the trained agent")
    evalMenuShowRun.pack()
    evalMenuButton = tk.Button(evalMenu, text="Go", command=lambda: showAgentRun(trained_agent))
    evalMenuButton.pack()

    evalMenuGraphsLabel = tk.Label(evalMenu, text="Show training graphs")
    evalMenuGraphsLabel.pack()
    evalMenuGraphsButton = tk.Button(evalMenu, text="Show Graphs", command=lambda: getGraphs(trained_agent))
    evalMenuGraphsButton.pack()

# %%
learning_rate = 0.01        # How fast to learn (higher = faster but less stable)
n_episodes = 100_000        # Number of hands to practice
start_epsilon = 1.0         # Start with 100% random actions
epsilon_decay = start_epsilon / (n_episodes / 2)  # Reduce exploration over time
final_epsilon = 0.1         # Always keep some exploration


root = tk.Tk()
root.title("Blackjack Q-Learning Agent Training Features")
label = tk.Label(root, text="Please define the hyperparameters.")
label.pack()

# Create input fields for hyperparameters
learning_rate_label = tk.Label(root, text="Learning Rate:")
learning_rate_label.pack()
lrExplanation = tk.Label(root, text="(Higher = faster learning but less stable)")
lrExplanation.pack()
learning_rate_entry = tk.Entry(root)
learning_rate_entry.pack()
learning_rate_entry.insert(0, str(learning_rate))

n_episodes_label = tk.Label(root, text="Number of Episodes:")
n_episodes_label.pack()
episodesExplanation = tk.Label(root, text="(Total hands to train on)")
episodesExplanation.pack()
n_episodes_entry = tk.Entry(root)
n_episodes_entry.pack()
n_episodes_entry.insert(0, str(n_episodes))

start_epsilon_label = tk.Label(root, text="Start Epsilon:")
start_epsilon_label.pack()
startEpsilonExplanation = tk.Label(root, text="(Initial exploration rate)")
startEpsilonExplanation.pack()
start_epsilon_entry = tk.Entry(root)
start_epsilon_entry.pack()
start_epsilon_entry.insert(0, str(start_epsilon))

epsilon_decay_label = tk.Label(root, text="Epsilon Decay:")
epsilon_decay_label.pack()
epsilonDecayExplanation = tk.Label(root, text="(Rate of exploration decrease)")
epsilonDecayExplanation.pack()
epsilon_decay_entry = tk.Entry(root)
epsilon_decay_entry.pack()
epsilon_decay_entry.insert(0, str(epsilon_decay))

final_epsilon_label = tk.Label(root, text="Final Epsilon:")
final_epsilon_label.pack()
finalEpsilonExplanation = tk.Label(root, text="(Minimum exploration rate)")
finalEpsilonExplanation.pack()
final_epsilon_entry = tk.Entry(root)
final_epsilon_entry.pack()
final_epsilon_entry.insert(0, str(final_epsilon))

naturalBlackjackLabel = tk.Label(root, text="Natural Blackjack:")
naturalBlackjackLabel.pack()
naturalBlackjackExplanation = tk.Label(root, text="(Whether to reward starting with a natural blackjack)")
naturalBlackjackExplanation.pack()
naturalBlackjackVar = tk.BooleanVar()
naturalBlackjackCheckbox = tk.Checkbutton(root, variable=naturalBlackjackVar)
naturalBlackjackCheckbox.pack()

trainButton = tk.Button(root, text="Train Agent", command=createTrainingWindow)
trainButton.pack()

root.mainloop()
