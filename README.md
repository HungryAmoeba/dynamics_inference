# Dynamics Inference

This repo will hopefully be of use for:

1. simulation of interacting dynamical systems
2. inference of dynamics from (simulated) observational data

## Usage

Clone the repo and change directories to it.

### 1. Install and Activate the Conda Environment

First, create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate sde_inf  # Replace 'my_env' with the actual environment name from environment.yml
```

### 2. Run `main.py`

Once the environment is set up, you can execute the main script:

```bash
python main.py
```

By default this runs the swamalator and visualizes the results with blender.

### 3. Using Hydra for Configuration

This project uses [Hydra](https://hydra.cc/) for managing configurations. You can override default arguments as follows:

```bash
python main.py param1=value1 param2=value2
```

To use a different configuration file:

```bash
python main.py --config-name=my_config
```

For hierarchical overrides:

```bash
python main.py +new_param=value
```

### 4. Example Run

```bash
python main.py dynamics=gravitation
```

I'll write some documentation once the functionality of the repo has been fleshed out. 

### 5. Advanced visualization

Install [Blender](https://www.blender.org) for visualization.
Download assets into a folder ./assets
I'll write documentation for this once its a bit more complete!

## Additional Information

- Ensure all dependencies are installed before running.
- Refer to `config.yaml` for available parameters.
- For debugging, use `HYDRA_FULL_ERROR=1` to get detailed error messages.

```bash
HYDRA_FULL_ERROR=1 python main.py
```

## Dynamics

The dynamics implemented as of 3/31: 

High dimensional swarmalator from:
Yadav, A., J, K., Chandrasekar, V. K., Zou, W., Kurths, J., Senthilkumar, D. V. (2024): Exotic swarming dynamics of high-dimensional swarmalators. - Physical Review E, 109, 4, 044212.

Gravity 

## Dynamics inference methods

TODO


