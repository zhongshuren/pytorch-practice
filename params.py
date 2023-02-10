import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='train', type=str)

parser.add_argument('--dataset_dir', default='./dataset', type=str)
parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str)
parser.add_argument('--exp_dir', default='./experiments', type=str)
parser.add_argument('--save_checkpoint', default='False', type=bool)

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--train_sample_size', default=60000, type=int)
parser.add_argument('--eval_sample_size', default=600, type=int)


parser.add_argument('--lr', default='0.1', type=float)
parser.add_argument('--momentum', default='0.9', type=float)
parser.add_argument('--weight_decay', default='1e-4', type=float)
