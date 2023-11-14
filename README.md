# Tiramisu-RL : Optimizing Tiramisu Code Using Reinforcement Learning

## Installation

To use this project, you'll need to do the following:

1. Clone the repository to your local machine.
2. Make sure you have Anaconda installed.
3. copy `config/config.yaml.example` to `config/config.yaml` to have your own config
4. Update the paths in the `config/config.yaml` file to match your preferences.

## Usage

To run the project, do the following:

1. Activate the conda environment:
`conda activate <tiramisu_env>`
2. Use `TiramisuEnvAPI()` to do the following : select a program, apply a transformation on the program, get the speedup of the schedule and the representation.
3. Open `tiramisu_api_tutorial.py` to see some examples of applying loop transformations.
4. You can find the code of the reinforcement learning agent+environment under `rl_agent/`

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and test them thoroughly.
4. Submit a pull request.
