from __future__ import annotations

import numpy as np
import torch


def draw_data(args, config, generator):
    args = np.asarray(args, dtype=np.float32)
    if args.ndim != 2:
        raise ValueError(f"Expected args with shape (n_samples, n_dimensions), got {args.shape}.")
    noise = generator.normal(
        loc=0.0,
        scale=float(config["data"]["std_dev"]),
        size=args.shape,
    ).astype(np.float32)
    return args + noise


def get_data(prior_sampler, config, generator, show_output=True):
    data_config = config["data"]
    total_data_points = (
        data_config["n_train"] + data_config["n_test"] + data_config["n_validation"]
    )
    n_class = total_data_points // 2

    labels = np.hstack((np.zeros(n_class), np.ones(n_class)))

    parameters_all = prior_sampler((3 * n_class,))
    parameters_a = parameters_all[:n_class, :]
    parameters_b = parameters_all[n_class : 2 * n_class, :]
    parameters_c = parameters_all[2 * n_class :, :]
    parameters = np.vstack((parameters_a, parameters_c))

    data_0 = draw_data(parameters_b, config, generator)
    data_1 = draw_data(parameters_c, config, generator)
    data = np.vstack((data_0, data_1))

    idx = np.arange(2 * n_class)
    generator.shuffle(idx)
    data, labels, parameters = [
        torch.tensor(x[idx], dtype=torch.float32) for x in (data, labels, parameters)
    ]

    n_train = data_config["n_train"]
    n_test = data_config["n_test"]
    n_validation = data_config["n_validation"]
    all_information = (data, labels, parameters)
    data_train, labels_train, parameters_train = [x[:n_train] for x in all_information]
    data_test, labels_test, parameters_test = [
        x[n_train : n_train + n_test] for x in all_information
    ]
    data_validation, labels_validation, parameters_validation = [
        x[n_train + n_test : n_train + n_test + n_validation] for x in all_information
    ]

    if show_output:
        print(f"Train: {data_train.shape}, {labels_train.shape}, {parameters_train.shape}")
        print(f"Test: {data_test.shape}, {labels_test.shape}, {parameters_test.shape}")
        print(
            "Validation: "
            f"{data_validation.shape}, {labels_validation.shape}, {parameters_validation.shape}"
        )

    return (
        (data_train, parameters_train, labels_train),
        (data_validation, parameters_validation, labels_validation),
        (data_test, parameters_test, labels_test),
    )
