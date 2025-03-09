# POMDPGym
Wrapper for <a href="https://gymnasium.farama.org/">Gymnasium</a> environments for reinforcement learning to work with POMDPs.jl. Includes options to get the observation space from pixels.

## Installation

* Install this package by opening julia and running `]add https://github.com/ancorso/POMDPGym.jl`.
* The Python dependencies <a href="https://gymnasium.farama.org/">gymnasium</a> and `pygame` will be automatically installed during the build step of this package.

### Atari and other environments
Currently, the automatic installation using `Conda.jl` does not install the Atari environments of Gymnasium. To do this, install Atari environments in a custom Python environment manually and ask `PyCall.jl` to use it. To elaborate, create a new Python virtual environment and run
```
pip install gymnasium[classic-control] gymnasium[atari] pygame
pip install autorom
```
Then run the shell command `AutoROM` and accept the Atari ROM license. Now you can configure `PyCall.jl` to use your Python environment following the <a href="https://github.com/JuliaPy/PyCall.jl?tab=readme-ov-file#specifying-the-python-version">instructions here</a>.

Optionally, you can also install <a href="http://www.mujoco.org/">MuJoCo</a>.

## Maintainer

Anthony Corso (acorso@stanford.edu)
