import argparse
import os
from torch.utils.data import DataLoader
# from datasets import Causal3DDataset, InterventionalPongDataset, VoronoiDataset, PinballDataset, BallInBoxesDataset
from datasets import InterventionalPongDataset
from models.model import CausalModel
import torch
import wandb
import numpy as np


DATASET_NAMES = ["ball_in_boxes", "pong", "causal3d", "pinball", "voronoi"]

def main(args):
    # if args.dataset_name not in DATASET_NAMES:
    #     raise ValueError(f"Datset name {args.dataset_name} not found!")

    # if 'ball_in_boxes' == args.dataset_name:
    #     DataClass = BallInBoxesDataset
    # elif 'pong' == args.dataset_name:
    #     DataClass = InterventionalPongDataset
    # elif 'causal3d' == args.dataset_name:
    #     DataClass = Causal3DDataset
    # elif 'voronoi' == args.dataset_name:
    #     DataClass = VoronoiDataset
    # elif 'pinball' == args.dataset_name:
    #     DataClass = PinballDataset

    DataClass = InterventionalPongDataset
    
    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=True, return_latents=True, 
        coarse_vars=False, img_width=args.img_width)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=True, return_latents=True,
        causal_vars=train_dataset.target_names(), coarse_vars=False, img_width=args.img_width)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)

    

    checkpoint_dir = os.path.join(args.output_dir, "causal_model", args.dataset_name, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.c_in = train_dataset.get_inp_channels()
    args.img_width = train_dataset.get_img_width()
    model = CausalModel(causal_var_info=train_dataset.get_causal_var_info(), 
                        img_width=args.img_width, 
                        c_in=args.c_in, 
                        c_hid=args.c_hid, 
                        is_mlp=False, 
                        device=device,
                        angle_reg_weight=args.angle_reg_weight,
                        checkpoint_dir=checkpoint_dir)

    # Training
    max_iters = args.num_epochs * len(train_loader)
    model.create_optimizer(lr=args.lr, weight_decay=args.weight_decay, warmup=args.warmup, max_iters=max_iters)
    model.train(args.num_epochs, train_loader, val_loader)


    # Evaluation
    train_avg_loss, train_avg_norm_dist = model.evaluate(train_loader)
    val_avg_loss, val_avg_norm_dist = model.evaluate(val_loader)


    print(f'final_train_avg_loss {train_avg_loss:.4f} | final_train_avg_norm_dist {train_avg_norm_dist:.4f} | final_val_avg_loss {val_avg_loss:.4f} | final_val_avg_norm_dist {val_avg_norm_dist:.4f}')
    wandb.log({f'final_train_avg_loss': train_avg_loss})
    wandb.log({f'final_train_avg_norm_dist': train_avg_norm_dist})
    wandb.log({f'final_test_avg_loss': val_avg_loss})
    wandb.log({f'final_test_avg_norm_dist': val_avg_norm_dist})



def get_args_parser():
    parser = argparse.ArgumentParser('CITRIS-CAUSAL-MODEL', add_help=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset_name', type=str, default="pong")
    parser.add_argument('--img_width', type=int, default=32)
    parser.add_argument('--c_in', type=int, default=3)
    parser.add_argument('--c_hid', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--angle_reg_weight', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--project_name', type=str, default='pgm_citris_vae')
    parser.add_argument('--data_dir', type=str, default='/home/mila/a/arefinmr/scratch/PGM/data/interventional_pong')
    parser.add_argument('--output_dir', type=str, default='/home/mila/a/arefinmr/scratch/PGM/output')

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