import os
import torch
from PIL import Image
import numpy as np
from utils import load_checkpoint


def test(model, args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.cuda()
    else:
        device = torch.device('cpu')

    state_dict = load_checkpoint(args)
    model.load_state_dict(state_dict)

    exp_dir = args.exp_dir
    test_image_paths = os.listdir(exp_dir)

    model.eval()
    for path in test_image_paths:
        image = Image.open(f'{exp_dir}/{path}')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).reshape(1, 1, 28, 28)
        image = image.to(device)
        out = model(image)
        _, pred = out.max(1)
        print(path, pred.item())
    pass
