import numpy as np
import torch, json
from pathlib import Path
try:
    from scipy.special import gamma as gamma_func
except ImportError:
    print("Scipy not found. Gamma distribution will not be available.")
    #from math import gamma as gamma_func

generator = np.random.default_rng(0)

def normalize(function, lower, upper):
    x = np.linspace(lower, upper, 1000)
    integral = np.trapezoid(function(x), x)
    def result_fct(z): return function(z) / integral
    result_fct.__name__ = function.__name__
    return result_fct

def uniform(x, min_val=0, max_val=10):
    return 1 / (max_val - min_val) if min_val <= x <= max_val else 0.

def normal(x, mean=5, std=4):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

def exponential(x, lam=.3):
    return lam * np.exp(-lam * x) if x >= 0 else 0.

def gamma(x, shape=2, scale=2):
    return (x ** (shape - 1) * np.exp(-x / scale)) / (scale ** shape * gamma_func(shape)) if x >= 0 else 0.

def sinusoidal(x, frequency=3, offset=5):
    return np.sin(frequency * x) + offset

def sample_from_pdf(pdf, n_samples, x_min, x_max):
    x_values = np.linspace(x_min, x_max, int(1e4))
    pdf_values = pdf(x_values)
    cdf_values = np.cumsum(pdf_values)
    cdf_values /= cdf_values[-1]

    random_values = generator.uniform(0, 1, n_samples)
    sampled_x = np.interp(random_values, cdf_values, x_values)
    return sampled_x

def get_data(prior, config, show_output=True):
    data_config = config["data"]
    total_data_points = data_config["n_train"] + data_config["n_test"] + data_config["n_validation"]
    n_class = total_data_points // 2

    #labels
    labels = np.hstack((np.zeros(n_class), np.ones(n_class)))

    #parameters
    #parameters_all = generator.uniform(low=data_config["parameter_range"][0], high=data_config["parameter_range"][1], size=3*n_class)
        
    parameters_all = sample_from_pdf(prior, 3*n_class, data_config["parameter_range"][0], data_config["parameter_range"][1])
    parameters_dependent = parameters_all[:n_class]
    parameters_independent = parameters_all[n_class:2*n_class]
    parameters_random = parameters_all[2*n_class:]
    parameters = np.hstack((parameters_independent, parameters_dependent))

    #data
    data_0 = generator.normal(loc=parameters_random, scale=data_config["std"], size=n_class)
    data_1 = generator.normal(loc=parameters_dependent, scale=data_config["std"], size=n_class)
    data = np.hstack((data_0, data_1))

    #shuffle:
    idx = np.arange(2*(n_class))
    generator.shuffle(idx)
    data, labels, parameters = [torch.tensor(x[idx]).float() for x in (data, labels, parameters)]

    #split:
    n_train, n_test, n_validation = data_config["n_train"], data_config["n_test"], data_config["n_validation"]
    all_information = (data, labels, parameters)
    data_train, labels_train, parameters_train, = [x[:n_train] for x in all_information]
    data_test, labels_test, parameters_test, = [x[n_train:n_train+n_test] for x in all_information]
    data_validation, labels_validation, parameters_validation, = [x[n_train+n_test:n_train+n_test+n_validation] for x in all_information]

    if show_output:
        print(f"Train: {data_train.shape}, {labels_train.shape}, {parameters_train.shape}")
        print(f"Test: {data_test.shape}, {labels_test.shape}, {parameters_test.shape}")
        print(f"Validation: {data_validation.shape}, {labels_validation.shape}, {parameters_validation.shape}")
    
    return (data_train, parameters_train, labels_train), (data_validation, parameters_validation, labels_validation), (data_test, parameters_test, labels_test)

def get_filepath(filename, config, root="figs"):
    """
    Return figs/config_XXX/filename where:
        - XXX is a zero-padded counter (starting at 000).
    Reuse existing directory if its config.json matches the given config; do not rewrite config.json.
    Create a new directory (and save config.json) only if no matching config is found.
    Handles filenames with subdirectories (e.g., "images/img.png").
    """
    root_path = Path(root)
    root_path.mkdir(exist_ok=True)
    # Canonical JSON for comparison
    config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
    # Try to find an existing config_* with identical config.json
    for existing_dir in sorted(root_path.glob("config_*")):
        parts = existing_dir.name.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            config_file = existing_dir / "config.json"
            if config_file.exists():
                try:
                    stored_str = json.dumps(json.loads(config_file.read_text()), sort_keys=True, separators=(',', ':'))
                    if stored_str == config_str:
                        # Reuse this directory; do not rewrite config.json
                        filepath = existing_dir / filename
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        return filepath
                except Exception:
                    # Skip unreadable or invalid JSON files
                    pass

    # No matching config found: create next config_XXX
    used_numbers = []
    for d in root_path.glob("config_*"):
        parts = d.name.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            used_numbers.append(int(parts[1]))
    next_num = (max(used_numbers) + 1) if used_numbers else 0
    new_dir = root_path / f"config_{next_num:03d}"
    new_dir.mkdir(parents=True, exist_ok=True)
    (new_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    filepath = new_dir / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    return filepath
