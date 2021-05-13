#!/usr/bin/env python
"""Script used to train the network for representing a 3D obkect as a set of
primitives
"""
import argparse
# from itertools import izip
import json
import random
import os
import string
import sys

import numpy as np
import torch

from arguments import add_voxelizer_parameters, add_nn_parameters,\
    add_dataset_parameters, add_training_parameters,\
    add_regularizer_parameters, add_sq_mesh_sampler_parameters,\
    add_gaussian_noise_layer_parameters, voxelizer_shape,\
    add_loss_options_parameters, add_loss_parameters, get_loss_options
from output_logger import get_logger
from utils import parse_train_test_splits

from learnable_primitives.common.dataset import get_dataset_type, \
    compose_transformations
from learnable_primitives.common.model_factory import DatasetBuilder
from learnable_primitives.common.batch_provider import BatchProviderPair
from learnable_primitives.equal_distance_sampler_sq import get_sampler
from learnable_primitives.models import NetworkParameters, train_on_batch_pair, \
    optimizer_factory
from learnable_primitives.loss_functions import euclidean_dual_loss_pair
from learnable_primitives.voxelizers import VoxelizerFactory


def moving_average(prev_val, new_val, b):
    return (prev_val*b + new_val) / (b+1)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))


def yield_infinite(iterable):
    while True:
        for item in iterable:
            yield item


def lr_schedule(optimizer, current_epoch, init_lr, factor, reductions):
    def inner(epoch):
        for i, e in enumerate(reductions):
            if epoch < e:
                return init_lr*factor**(-i)
        return init_lr*factor**(-len(list(reductions)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = inner(current_epoch)

    return optimizer


def get_weight(w, epoch, current_epoch):
    if current_epoch < epoch:
        return 0.0
    else:
        return w


def get_regularizer_terms(args, current_epoch):
    # Create a dictionary with the regularization terms if there are any
    regularizer_terms = {
        "regularizer_type": args.regularizer_type,
        "bernoulli_regularizer_weight": args.bernoulli_regularizer_weight,
        "entropy_bernoulli_regularizer_weight":
            args.entropy_bernoulli_regularizer_weight,
        "parsimony_regularizer_weight": args.parsimony_regularizer_weight,
        "sparsity_regularizer_weight": args.sparsity_regularizer_weight,
        "overlapping_regularizer_weight": args.overlapping_regularizer_weight,
        "minimum_number_of_primitives": args.minimum_number_of_primitives,
        "maximum_number_of_primitives": args.maximum_number_of_primitives,
        "w1": args.w1,
        "w2": args.w2
    }

    return regularizer_terms


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(t[k]) for k in t}

    git_head_hash = "foo"
    params["git-commit"] = git_head_hash
    params["experiment_tag"] = experiment_tag
    for k, v in params.items():
        if v == "":
            params[k] = None
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Train a network to predict primitives"
    )
    parser.add_argument(
        "dataset_directory",
        help="Path to the directory containing the dataset"
    )
    parser.add_argument(
        "output_directory",
        help="Save the output files in that directory"
    )

    parser.add_argument(
        "--tsdf_directory",
        default="",
        help="Path to the directory containing the precomputed tsdf files"
    )
    parser.add_argument(
        "--weight_file",
        default=None,
        help=("The path to a previously trainined model to continue"
              " the training from")
    )
    parser.add_argument(
        "--continue_from_epoch",
        default=0,
        type=int,
        help="Continue training from epoch (default=0)"
    )
    parser.add_argument(
        "--n_primitives",
        type=int,
        default=32,
        help="Number of primitives"
    )
    parser.add_argument(
        "--use_deformations",
        action="store_true",
        help="Use Superquadrics with deformations as the shape configuration"
    )
    parser.add_argument(
        "--train_test_splits_file",
        default=None,
        help="Path to the train-test splits file"
    )
    parser.add_argument(
        "--run_on_gpu",
        action="store_true",
        help="Use GPU"
    )
    parser.add_argument(
        "--probs_only",
        action="store_true",
        help="Optimize only using the probabilities"
    )

    parser.add_argument(
        "--experiment_tag",
        default=None,
        help="Tag that refers to the current experiment"
    )

    parser.add_argument(
        "--cache_size",
        type=int,
        default=2000,
        help="The batch provider cache size"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Seed for the PRNG"
    )

    add_nn_parameters(parser)
    add_dataset_parameters(parser)
    add_voxelizer_parameters(parser)
    add_training_parameters(parser)
    add_sq_mesh_sampler_parameters(parser)
    add_regularizer_parameters(parser)
    add_gaussian_noise_layer_parameters(parser)
    # Parameters related to the loss function and the loss weights
    add_loss_parameters(parser)
    # Parameters related to loss options
    add_loss_options_parameters(parser)
    args = parser.parse_args(argv)

    if args.train_test_splits_file is not None:
        train_test_splits = parse_train_test_splits(
            args.train_test_splits_file,
            args.model_tags
        )
        training_tags = np.hstack([
            train_test_splits["train"],
            train_test_splits["val"]
        ])
    else:
        training_tags = args.model_tags

    #device = torch.device("cuda:0")
    if args.run_on_gpu: #and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Running code on {}".format(device))

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    # Store the parameters for the current experiment in a json file
    save_experiment_params(args, experiment_tag, experiment_directory)
    print("Save experiment statistics in %s" %(experiment_tag, ))

    # Create two files to store the training and test evolution
    train_stats = os.path.join(experiment_directory, "train.txt")
    val_stats = os.path.join(experiment_directory, "val.txt")
    if args.weight_file is None:
        train_stats_f = open(train_stats, "w")
    else:
        train_stats_f = open(train_stats, "a+")
    train_stats_f.write((
        "epoch loss pcl_to_prim_loss prim_to_pcl_loss bernoulli_regularizer "
        "entropy_bernoulli_regularizer parsimony_regularizer "
        "overlapping_regularizer sparsity_regularizer sparse_probability_change parameter_change \n"
    ))

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    # Create an object that will sample points in equal distances on the
    # surface of the primitive
    sampler = get_sampler(
        args.use_cuboids,
        args.n_points_from_sq_mesh,
        args.D_eta,
        args.D_omega
    )

    # Create a factory that returns the appropriate voxelizer based on the
    # input argument
    voxelizer_factory = VoxelizerFactory(
        args.voxelizer_factory,
        np.array(voxelizer_shape(args)),
        args.save_voxels_to
    )

    # Create a dataset instance to generate the samples for training
    training_dataset = get_dataset_type("euclidean_dual_loss_pair")(
        (DatasetBuilder()
            .with_dataset(args.dataset_type)
            .lru_cache(2000)
            .filter_tags(training_tags)
            .build(args.dataset_directory)),
        '/orion/u/ianhuang/superquadric_parsing/chair_triplets.npz',
        voxelizer_factory,
        args.n_points_from_mesh,
        transform=compose_transformations(args.voxelizer_factory)
    )

    # Create a batchprovider object to start generating batches
    train_bp = BatchProviderPair(
        training_dataset,
        batch_size=args.batch_size,
        cache_size=args.cache_size
    )
    train_bp.ready()

    network_params = NetworkParameters.from_options(args)
    # Build the model to be used for training
    model = network_params.network(network_params)

    # Move model to the device to be used
    model.to(device)
    # Check whether there is a weight file provided to continue training from
    if args.weight_file is not None:
        model.load_state_dict(torch.load(args.weight_file))
    model.train()

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(args, model)

    # Loop over the dataset multiple times
    pcl_to_prim_losses = []
    prim_to_pcl_losses = []
    losses = []


    for i in range(args.epochs):
        bar = get_logger(
            "euclidean_dual_loss_pair",
            i+1,
            args.epochs,
            args.steps_per_epoch
        )
        for b, sample in zip(range(args.steps_per_epoch), yield_infinite(train_bp)):

            X_from, y_target_from, X_to, y_target_to = sample

            X_from = X_from.to(device)
            X_to = X_to.to(device)
            y_target_from = y_target_from.to(device)
            y_target_to = y_target_to.to(device)

            batch_loss, metrics, debug_stats = train_on_batch_pair(
                model,
                lr_schedule(
                    optimizer, i, args.lr, args.lr_factor, args.lr_epochs
                ),
                euclidean_dual_loss_pair,
                X_from,
                X_to,
                y_target_from,
                y_target_to,
                get_regularizer_terms(args, i),
                sampler,
                get_loss_options(args)
            )

            delta_prob_L1 = debug_stats["delta_prob_L1"].item()
            delta_params_L2 = debug_stats["delta_params_L2"].item()
            print("delta_prob_L1: {}   delta_params_L2: {}".format(delta_prob_L1, delta_params_L2))

            # Get the regularizer terms
            reg_values_from = debug_stats["regularizer_terms_from"]
            sparsity_regularizer_from = reg_values_from["sparsity_regularizer"]
            overlapping_regularizer_from = reg_values_from ["overlapping_regularizer"]
            parsimony_regularizer_from = reg_values_from["parsimony_regularizer"]
            entropy_bernoulli_regularizer_from = reg_values_from["entropy_bernoulli_regularizer"]
            bernoulli_regularizer_from = reg_values_from["bernoulli_regularizer"]
            # The lossess
            pcl_to_prim_loss_from = debug_stats["pcl_to_prim_loss_from"].item()
            prim_to_pcl_loss_from = debug_stats["prim_to_pcl_loss_from"].item()


            # Get the regularizer terms
            reg_values_to = debug_stats["regularizer_terms_to"]
            sparsity_regularizer_to = reg_values_to["sparsity_regularizer"]
            overlapping_regularizer_to = reg_values_to["overlapping_regularizer"]
            parsimony_regularizer_to = reg_values_to["parsimony_regularizer"]
            entropy_bernoulli_regularizer_to = reg_values_to["entropy_bernoulli_regularizer"]
            bernoulli_regularizer_to = reg_values_to["bernoulli_regularizer"]
            # The lossess
            pcl_to_prim_loss_to = debug_stats["pcl_to_prim_loss_to"].item()
            prim_to_pcl_loss_to = debug_stats["prim_to_pcl_loss_to"].item()


            pcl_to_prim_loss = pcl_to_prim_loss_from + pcl_to_prim_loss_to
            prim_to_pcl_loss = prim_to_pcl_loss_from + prim_to_pcl_loss_to
            sparsity_regularizer = sparsity_regularizer_from + sparsity_regularizer_to
            overlapping_regularizer = overlapping_regularizer_from + overlapping_regularizer_to
            parsimony_regularizer = parsimony_regularizer_from + parsimony_regularizer_to
            entropy_bernoulli_regularizer = entropy_bernoulli_regularizer_from + entropy_bernoulli_regularizer_to
            bernoulli_regularizer = bernoulli_regularizer_from + bernoulli_regularizer_to

            bar.loss = moving_average(bar.loss, batch_loss, b)
            bar.pcl_to_prim_loss = \
                moving_average(bar.pcl_to_prim_loss, pcl_to_prim_loss, b)
            bar.prim_to_pcl_loss = \
                moving_average(bar.prim_to_pcl_loss, prim_to_pcl_loss, b)

            losses.append(bar.loss)
            prim_to_pcl_losses.append(bar.prim_to_pcl_loss)
            pcl_to_prim_losses.append(bar.pcl_to_prim_loss)

            bar.bernoulli_regularizer =\
                (bar.bernoulli_regularizer * b + bernoulli_regularizer) / (b+1)
            bar.parsimony_regularizer =\
                (bar.parsimony_regularizer * b + parsimony_regularizer) / (b+1)
            bar.overlapping_regularizer =\
                (bar.overlapping_regularizer * b + overlapping_regularizer) / (b+1)
            bar.entropy_bernoulli_regularizer = \
                (bar.entropy_bernoulli_regularizer * b +
                 entropy_bernoulli_regularizer) / (b+1)
            bar.sparsity_regularizer =\
                (bar.sparsity_regularizer * b + sparsity_regularizer) / (b+1)

            bar.exp_n_prims = metrics[0].sum(-1).mean()
            # Update the file that keeps track of the statistics
            train_stats_f.write(
                ("%d %.8f %.8f %.8f %.6f %.6f %.6f %.6f %.6f %.6f %.6f") %(
                i, bar.loss, bar.pcl_to_prim_loss, bar.prim_to_pcl_loss,
                bar.bernoulli_regularizer,
                bar.entropy_bernoulli_regularizer,
                bar.parsimony_regularizer,
                bar.overlapping_regularizer,
                bar.sparsity_regularizer,
                delta_prob_L1,
                delta_params_L2
                )
            )
            train_stats_f.write("\n")
            train_stats_f.flush()

            bar.next()
        # Finish the progress bar and save the model after every epoch
        bar.finish()
        # Stop the batch provider
        train_bp.stop()
        torch.save(
            model.state_dict(),
            os.path.join(
                experiment_directory,
                "model_%d" % (i + args.continue_from_epoch,)
            )
        )

    print ([
        sum(losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(losses[:args.steps_per_epoch]) / float(args.steps_per_epoch),
        sum(pcl_to_prim_losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(pcl_to_prim_losses[:args.steps_per_epoch]) / float(args.steps_per_epoch),
        sum(prim_to_pcl_losses[args.steps_per_epoch:]) / float(args.steps_per_epoch),
        sum(prim_to_pcl_losses[:args.steps_per_epoch]) / float(args.steps_per_epoch),
    ])


if __name__ == "__main__":
    main(sys.argv[1:])
