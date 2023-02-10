import torch
from matplotlib import pyplot as plt


def save_checkpoint(model, args):
    torch.save(model.state_dict(), f'{args.checkpoint_dir}/checkpoint.pth')
    pass


def load_checkpoint(args):
    return torch.load(f'{args.checkpoint_dir}/checkpoint.pth')


def plot_tensor(img, title):
    plt.imshow(img.cpu().numpy())
    plt.title(title)
    plt.show()
    pass
