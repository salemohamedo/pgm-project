import argparse
import torch.utils.data as data
from training_module import CITRISVAE
# from experiments.utils import train_model, load_datasets, print_params
import os
import argparse
import json
import torch 
import torch.utils.data as data
import pytorch_lightning as pl 
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import InterventionalPongDataset
import copyfile


def get_device():
    return torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def load_datasets(args):
    pl.seed_everything(args.seed)
    print('Loading datasets...')
    data_name = 'pong'
    DataClass = InterventionalPongDataset
    dataset_args = {}

    test_args = lambda train_set: {'causal_vars': train_set.target_names_l}
    train_dataset = InterventionalPongDataset(
        data_folder=args.data_dir, split='train', single_image=False, triplet=False, seq_len=args.seq_len, **dataset_args)
    val_dataset = InterventionalPongDataset(
        data_folder=args.data_dir, split='val_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
    val_triplet_dataset = InterventionalPongDataset(
        data_folder=args.data_dir, split='val', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
    test_dataset = InterventionalPongDataset(
        data_folder=args.data_dir, split='test_indep', single_image=True, triplet=False, return_latents=True, **dataset_args, **test_args(train_dataset))
    test_triplet_dataset = InterventionalPongDataset(
        data_folder=args.data_dir, split='test', single_image=False, triplet=True, return_latents=True, **dataset_args, **test_args(train_dataset))
    if args.exclude_objects is not None and data_name == 'causal3d':
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
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_triplet_loader = data.DataLoader(val_triplet_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_triplet_loader = data.DataLoader(test_triplet_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)

    print(f'Training dataset size: {len(train_dataset)} / {len(train_loader)}')
    # print(f'Val triplet dataset size: {len(val_triplet_dataset)} / {len(val_triplet_loader)}')
    if isinstance(val_dataset, dict):
        print(f'Val correlation dataset sizes: { {key: len(val_dataset[key]) for key in val_dataset} }')
    else:
        print(f'Val correlation dataset size: {len(val_dataset)}')
    # print(f'Test triplet dataset size: {len(test_triplet_dataset)} / {len(test_triplet_loader)}')
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
    return datasets, data_loaders, data_name

def print_params(logger_name, model_args):
    num_chars = max(50, 11+len(logger_name))
    print('=' * num_chars)
    print(f'Experiment {logger_name}')
    print('-' * num_chars)
    for key in sorted(list(model_args.keys())):
        print(f'-> {key}: {model_args[key]}')
    print('=' * num_chars)

def train_model(model_class, train_loader, val_loader, 
                test_loader=None,
                logger_name=None,
                max_epochs=200,
                progress_bar_refresh_rate=1,
                check_val_every_n_epoch=1,
                debug=False,
                offline=False,
                op_before_running=None,
                load_pretrained=False,
                root_dir=None,
                files_to_save=None,
                gradient_clip_val=1.0,
                cluster=False,
                callback_kwargs=None,
                seed=42,
                save_last_model=False,
                val_track_metric='val_loss',
                data_dir=None,
                **kwargs):
    trainer_args = {}
    if root_dir is None or root_dir == '':
        root_dir = os.path.join('checkpoints/', model_class.__name__)
    if not (logger_name is None or logger_name == ''):
        logger_name = logger_name.split('/')
        logger = pl.loggers.TensorBoardLogger(root_dir, 
                                              name=logger_name[0], 
                                              version=logger_name[1] if len(logger_name) > 1 else None)
        trainer_args['logger'] = logger
    if progress_bar_refresh_rate == 0:
        trainer_args['enable_progress_bar'] = False

    if callback_kwargs is None:
        callback_kwargs = dict()
    callbacks = model_class.get_callbacks(exmp_inputs=next(iter(val_loader)), cluster=cluster, 
                                          **callback_kwargs)

    print("here")
    print(root_dir)
    if not debug:
        callbacks.append(
                ModelCheckpoint(save_weights_only=True, 
                                mode="min", 
                                monitor=val_track_metric,
                                save_last=save_last_model,
                                every_n_epochs=check_val_every_n_epoch)
            )
    if debug:
        torch.autograd.set_detect_anomaly(True) 
    trainer = pl.Trainer(default_root_dir=root_dir,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=max_epochs,
                         callbacks=callbacks,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         gradient_clip_val=gradient_clip_val,
                         **trainer_args)
    trainer.logger._default_hp_metric = None

    if files_to_save is not None:
        log_dir = trainer.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        for file in files_to_save:
            if os.path.isfile(file):
                filename = file.split('/')[-1]
                copyfile(file, os.path.join(log_dir, filename))
                print(f'=> Copied {filename}')
            else:
                print(f'=> File not found: {file}')

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(
        'checkpoints/', model_class.__name__ + ".ckpt")
    if load_pretrained and os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        if load_pretrained:
            print("Warning: Could not load any pretrained models despite", load_pretrained)
        pl.seed_everything(seed)  # To be reproducable
        model = model_class(**kwargs)
        if op_before_running is not None:
            op_before_running(model)
        trainer.fit(model, train_loader, val_loader)
        # model = model_class.load_from_checkpoint(
        #     trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    if test_loader is not None:
        # model_paths = [(trainer.checkpoint_callback.best_model_path, "best")]
        model_paths = []
        if save_last_model:
            model_paths += [(trainer.checkpoint_callback.last_model_path, "last")]
        for file_path, prefix in model_paths:
            model = model_class.load_from_checkpoint(file_path)
            for c in callbacks:
                if hasattr(c, 'set_test_prefix'):
                    c.set_test_prefix(prefix)
            test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
            test_result = test_result[0]
            print('='*50)
            print(f'Test results ({prefix}):')
            print('-'*50)
            for key in test_result:
                print(key + ':', test_result[key])
            print('='*50)

            log_file = os.path.join(trainer.logger.log_dir, f'test_results_{prefix}.json')
            with open(log_file, 'w') as f:
                json.dump(test_result, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Default arguments
    parser.add_argument('--data_dir', type=str, required=True)
    # parser.add_argument('--causal_encoder_checkpoint', type=str, required=True)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exclude_vars', type=str, nargs='+', default=None)
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None)
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--data_img_width', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--imperfect_interventions', action='store_true')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=5)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')

    parser.add_argument('--model', type=str, default='CITRISVAE')
    parser.add_argument('--c_hid', type=int, default=32)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--classifier_use_conditional_targets', action='store_true')
    parser.add_argument('--kld_warmup', type=int, default=0)
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0)
    parser.add_argument('--lambda_sparse', type=float, default=0.02)
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")

    args = parser.parse_args()
    model_args = vars(args)

    datasets, data_loaders, data_name = load_datasets(args)

    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['img_width'] = datasets['train'].get_img_width()
    model_args['num_causal_vars'] = datasets['train'].num_vars()
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])
    if hasattr(datasets['train'], 'get_inp_channels'):
        model_args['c_in'] = datasets['train'].get_inp_channels()
    model_name = model_args.pop('model')
    model_class = CITRISVAE

    logger_name = f'{model_name}_{args.num_latents}l_{model_args["num_causal_vars"]}b_{args.c_hid}hid_{data_name}'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)
    
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 2 if not args.cluster else 25

    print(datasets['train'])
    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val_triplet'],
                test_loader=data_loaders['test_triplet'],
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs={'dataset': datasets['train'], 
                                 'correlation_dataset': datasets['val'],
                                 'correlation_test_dataset': datasets['test']},
                var_names=datasets['train'].target_names(),
                save_last_model=True,
                cluster_logging=args.cluster,
                causal_var_info=datasets["train"].get_causal_var_info(),
                **model_args)
