import argparse
import json
import logging
import os
import random

import numpy as np
import torch

import wandb
from datasets.eth_xgaze import get_train_loader
from trainer.gazenerf_trainer import get_trainer
from utils.logging import config_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train options")
    add_arg = parser.add_argument
    add_arg("--gpu_id", type=int, default=0)
    add_arg("--batch_size", type=int, default=1)
    add_arg("--num_workers", type=int, default=0)
    add_arg("--num_epochs", type=int, default=100)
    add_arg("--num_iterations", type=int, default=1)
    add_arg("--step_decay", type=int, default=1000)
    add_arg("--learning_rate", type=float, default=0.0001)
    add_arg("--vgg_importance", type=float, default=1.0)
    add_arg("--eye_loss_importance", type=float, default=100.0)
    add_arg("--img_dir", type=str, default="data/eth_xgaze/subjects")
    add_arg("--bg_type", type=str, default="white")
    add_arg("--checkpoint_dir", type=str, default=None)
    add_arg("--optimizer", type=str, default="adam")
    add_arg("--state_dict_name", type=str, default="tmp.json")
    add_arg("--model_path", type=str, default="checkpoints/2_mlp.json")
    add_arg("--log", type=bool, default=False)
    add_arg("--resume", type=bool, default=True)
    add_arg("--verbose", type=bool, default=True)
    add_arg("--use_vgg_loss", type=bool, default=True)
    add_arg("--use_l1_loss", type=bool, default=True)
    add_arg("--use_angular_loss", type=bool, default=False)
    add_arg("--use_patch_gan_loss", type=bool, default=False)
    add_arg("--include_vd", type=bool, default=False)
    add_arg("--hier_sampling", type=bool, default=False)
    add_arg("--enable_ffhq", type=bool, default=False)
    add_arg("--enable_eth_xgaze", type=bool, default=True)
    add_arg("--fit_image", type=bool, default=False)
    return parser.parse_args()


def process(args, key, val=False):
    train_dataloader = get_train_loader(
        args.img_dir,
        args.batch_size,
        args.num_workers,
        is_shuffle=False,
        subject=key,
        evaluate="landmark",
    )

    # Load the trainer
    gpu = args.gpu_id
    if gpu is not None and torch.cuda.is_available():
        logging.info("Using GPU %i", gpu)

    fit_image_bool = args.fit_image

    if val:
        fit_image_bool = False

    trainer = get_trainer(
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        gpu=gpu,
        resume=args.resume,
        include_vd=args.include_vd,
        hier_sampling=args.hier_sampling,
        log=args.log,
        lr=args.learning_rate,
        num_iter=args.num_iterations,
        optimizer=args.optimizer,
        step_decay=args.step_decay,
        vgg_importance=args.vgg_importance,
        eye_loss_importance=args.eye_loss_importance,
        fit_image=fit_image_bool,
        model_path=args.model_path,
        state_dict_name=args.state_dict_name,
        use_vgg_loss=args.use_vgg_loss,
        use_l1_loss=args.use_l1_loss,
        use_angular_loss=args.use_angular_loss,
        use_patch_gan_loss=args.use_patch_gan_loss,
    )

    # Run the training
    trainer.train_single_image(train_dataloader, args.num_epochs, 0, "one_fit")
    trainer.evaluate_single_image(
        data_loader=train_dataloader,
        key=key,
        val=val,
    )


def main():
    """Main function to produce images with redirected gaze"""

    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Initialization
    args = parse_args()

    if args.log:
        wandb.init(project="evaluation", config={"gpu_id": 0})
        wandb.config.update(args)

    # Setup logging
    log_file = "logs"
    if not os.path.exists(log_file):
        os.makedirs(log_file)
    log_file = os.path.join(log_file, "out_%i.log" % args.gpu_id)
    config_logging(verbose=args.verbose, log_file=log_file, append=args.resume)

    refer_list_file = os.path.join("data/eth_xgaze", "train_test_split.json")

    with open(refer_list_file, "r") as f:
        datastore = json.load(f)

    train_keys = datastore["train"]
    val_keys = datastore["val"]

    for subject in train_keys:
        process(args, subject)

    #for subject in val_keys:
    #        process(args, subject)

    # for subject in val_keys:
    #    process(args, subject, True)


if __name__ == "__main__":
    main()
