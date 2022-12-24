import argparse
from torch.utils import data
import os
import argparse
import json
import torch 
from torch.utils.data import DataLoader
from datasets import Causal3DDataset, InterventionalPongDataset, VoronoiDataset
import numpy as np
import wandb
from models.model import CITRISVAE

DATASET_NAMES = ["pong","causal3d", "voronoi"] #["ball_in_boxes", "pong", "causal3d", "pinball", "voronoi"]

def load_datasets(args):
    if args.dataset_name not in DATASET_NAMES:
        raise ValueError(f"Datset name {args.dataset_name} not found!")

    if 'pong' == args.dataset_name:
        DataClass = InterventionalPongDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
    elif 'causal3d' == args.dataset_name:
        DataClass = Causal3DDataset
        dataset_args = {'coarse_vars': args.coarse_vars, 'exclude_vars': args.exclude_vars, 'exclude_objects': args.exclude_objects}
        test_args = lambda train_set: {'causal_vars': train_set.full_target_names}
    elif 'voronoi' == args.dataset_name:
        DataClass = VoronoiDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names_l}

    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=False, triplet=False, seq_len=args.seq_len, **dataset_args)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
    val_triplet_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
    test_dataset = DataClass(
        data_folder=args.data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
    test_triplet_dataset = DataClass(
        data_folder=args.data_dir, split='test', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
    if args.exclude_objects is not None and args.dataset_name == 'causal3d':
        test_dataset = {
            'orig_wo_' + '_'.join([str(o) for o in args.exclude_objects]): test_dataset
        }
        val_dataset = {
            next(iter(test_dataset.keys())): val_dataset 
        }
        dataset_args.pop('exclude_objects')
        for o in args.exclude_objects:
            val_dataset[f'exclusive_obj_{o}'] = DataClass(
                                data_folder=args.data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, exclude_objects=[i for i in range(7) if i != o], **dataset_args, **test_args(train_dataset))
            test_dataset[f'exclusive_obj_{o}'] = DataClass(
                                data_folder=args.data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, exclude_objects=[i for i in range(7) if i != o], **dataset_args, **test_args(train_dataset))
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=False, drop_last=True, num_workers=args.num_workers)
    val_triplet_loader = data.DataLoader(val_triplet_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_triplet_loader = data.DataLoader(test_triplet_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)

    print(f'Training dataset size: {len(train_dataset)} / {len(train_loader)}')
    print(f'Val triplet dataset size: {len(val_triplet_dataset)} / {len(val_triplet_loader)}')
    if isinstance(val_dataset, dict):
        print(f'Val correlation dataset sizes: { {key: len(val_dataset[key]) for key in val_dataset} }')
    else:
        print(f'Val correlation dataset size: {len(val_dataset)}')
    print(f'Test triplet dataset size: {len(test_triplet_dataset)} / {len(test_triplet_loader)}')
    if isinstance(test_dataset, dict):
        print(f'Test correlation dataset sizes: { {key: len(test_dataset[key]) for key in test_dataset} }')
    else:
        print(f'Test correlation dataset size: {len(test_dataset)}')
        

    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'val_triplet': val_triplet_dataset,
        'test': test_dataset,
        'test_triplet': test_triplet_dataset
    }
    data_loaders = {
        'train': train_loader,
        'val_triplet': val_triplet_loader,
        'test_triplet': test_triplet_loader
    }
    return datasets, data_loaders

def main(args):

    datasets, data_loaders = load_datasets(args)

    checkpoint_dir = os.path.join(args.output_dir, args.model_name, args.dataset_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logdir = os.path.join(args.output_dir, args.model_name, args.dataset_name, "log")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    args.logdir = logdir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.c_in = datasets["train"].get_inp_channels()
    args.img_width = datasets["train"].get_img_width()
    args.max_iters = args.num_epochs * len(data_loaders['train'])
    # args.warmup = args.warmup * len(data_loaders['train'])
    args.num_causal_vars = datasets["train"].num_vars()

    model = CITRISVAE(args, device)

    # Training
    model.train(data_loaders['train'], data_loaders['val_triplet'], datasets['val'], args.num_epochs, datasets['train'], checkpoint_dir)

    # Load the best model
    checkpoint = torch.load(os.path.join(checkpoint_dir, f"best.pt"))
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.intervention_classifier.load_state_dict(checkpoint['intervention_classifier'])
    model.transition_prior.load_state_dict(checkpoint['transition_prior'])
    if args.use_flow_prior:
        model.transition_prior.load_state_dict(checkpoint['flow'])

    # Evaluate with triplet on test data
    test_avg_loss, test_avg_norm_dist = model.evaluate_with_triplet(data_loaders['test_triplet'], split="test")
    wandb.log({f'triplet_test_avg_loss': test_avg_loss})
    wandb.log({f'triplet_test_avg_norm_dist': test_avg_norm_dist})

    # Evaluate correlation on test data
    model.evaluate_correlation(datasets["test"], split="test", logdir=logdir)


def get_args_parser():
    parser = argparse.ArgumentParser('CITRIS-CAUSAL-MODEL', add_help=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="pong")
    parser.add_argument('--model_name', type=str, default="citris_vae")
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--probe_num_epochs', type=int, default=100)
    parser.add_argument('--probe_lr', type=float, default=4e-3)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--img_width', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--imperfect_interventions', action='store_true')
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--use_flow_prior', action='store_true')
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--exclude_vars', type=str, nargs='+', default=None)
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None)
    parser.add_argument('--check_correlation_every_n_epoch', type=int, default=10)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--kld_warmup', type=int, default=0)
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--project_name', type=str, default='pgm_citris_vae')
    parser.add_argument('--data_dir', type=str, default='/home/mila/a/arefinmr/scratch/PGM/data/interventional_pong')
    parser.add_argument('--output_dir', type=str, default='/home/mila/a/arefinmr/scratch/PGM/output')
    parser.add_argument('--pretrained_causal_model_path', type=str, default='/home/mila/a/arefinmr/scratch/PGM/output/causal_checkpoints/pong_causal_checkpoint.ckpt')
    

    return parser


def init_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    init_seed(args)

    run = wandb.init(project=args.project_name, config=args, reinit=False, entity='pgm-project', settings=wandb.Settings(start_method="fork"))
    main(args)
    run.finish()
    print("Finish!!!!")
