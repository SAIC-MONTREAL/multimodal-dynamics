import os
import argparse

from mmdyn.pytorch import config
from mmdyn.pytorch.utils.training import save_pkl
from mmdyn.pytorch.problems.problems import Regression, Reconstruction, SeqModeling, DynModeling


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # Problem
    parser.add_argument('--problem-type', default='seq_modeling', type=str,
                        help='Problem type (default: seq_modeling)')
    parser.add_argument('--model-name', default='cnn-mvae', type=str,
                        help='Model architecture name')
    parser.add_argument('--input-type', default='visual', type=str,
                        help='The input modality (default: visuotactile) (valid: visual, tactile, visuotactile)')
    parser.add_argument('--use-pose', action='store_true', default=False,
                        help="Use pose as additional modality, only works for MVAE) (default: False)")
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--dataset-path', default="~/dataset", type=str,
                        help='Absolute path to the dataset.')
    parser.add_argument('--batchsize', default=128, type=int,
                        help='Batchsize (default: 128)')
    parser.add_argument('--criterion', default="crossentropy", type=str,
                        help='Training loss (default: crossentropy)')
    parser.add_argument('--optimizer', default="Adam", type=str,
                        help='Name of gradient descent algorithm as defined in pytorch (default: Adam)')
    parser.add_argument('--num-epochs', default=100, type=int,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--mask-loss', action='store_true', default=False,
                        help="Mask the reconstruction loss to the object segment (default: False)")
    parser.add_argument('--vis-pose', action='store_true', default=False,
                        help="Visualize pose (warning: very slow) (default: False)")
    parser.add_argument('--pose-multiplier', default=1000, type=float,
                        help="Multiplier for pose loss (default: 1000)")

    # Misc
    parser.add_argument('--save-name', default='run', type=str,
                        help='Name given to model used for saving checkpoints (default: run)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help="Do not use CUDA (default: False)")

    # VAE specific
    parser.add_argument('--kl-weight', type=float, default=1.0,
                        help="KL weight in the loss of VAE models (default: 1)")
    parser.add_argument('--latent-size', type=int, default=256,
                        help="Latent dimension (default: 256)")
    parser.add_argument('--annealing-epochs', type=int, default=50,
                        help="Number of epochs to anneal KL for (default: 50)")
    parser.add_argument('--conditional', action='store_true', default=False,
                        help="Use conditional VAE (useful for the force perturbation scenario) (default: False)")

    args = parser.parse_args()

    # Setup problem
    assert args.problem_type in config.PROBLEM_TYPES, "Invalid problem type."
    if args.problem_type == 'regression':
        problem = Regression(args)
    elif args.problem_type == 'reconstruction':
        problem = Reconstruction(args)
    elif args.problem_type == 'dyn_modeling':
        problem = DynModeling(args)
    else:
        problem = SeqModeling(args)

    save_pkl(args, os.path.join(problem.log_dir, 'problem.pkl'))

    problem.train()
