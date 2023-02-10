import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from dataset.dataloader import dataloader
from tqdm import tqdm
from utils import save_checkpoint


def train(model, args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.cuda()
    else:
        device = torch.device('cpu')

    train_data = dataloader(args, 'train')
    eval_data = dataloader(args, 'eval')
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=[args.epochs * 0.5, args.epochs * 0.8],
                            gamma=0.1)

    for epoch in range(args.epochs):
        info = {'lr': '%.1e' % scheduler.get_last_lr()[0],
                'avg_train_loss': '#.###',
                'avg_train_acc': '##.###',
                'avg_eval_loss': '#.###',
                'avg_eval_acc': '##.###'}
        p_bar = tqdm(desc=f'epoch {epoch + 1}/{args.epochs}',
                     total=len(train_data),
                     postfix=info,
                     bar_format='{l_bar}{bar}|\t[{n_fmt}/{total_fmt}{postfix}]')

        # start training
        train_loss = 0
        train_acc = 0
        model.train()
        for i, item in enumerate(train_data):
            # prepare input
            image = item['image'].to(device)
            label = item['label'].to(device)

            # forward propagation
            out = model(image)
            loss = criterion(out, label)
            _, pred = out.max(1)
            n_correct = pred.eq(label).float().mean()

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # visualize
            train_loss += loss.item()
            train_acc += n_correct.item()
            p_bar.update(1)
            info['avg_train_loss'] = '%.3f' % (train_loss / (i + 1))
            info['avg_train_acc'] = '%.3f' % (train_acc / (i + 1) * 100)
            p_bar.set_postfix(info)
        scheduler.step()

        # start evaluating
        eval_loss = 0
        eval_acc = 0
        model.eval()
        for i, item in enumerate(eval_data):
            # prepare input
            image = item['image'].to(device)
            label = item['label'].to(device)

            # forward propagation
            out = model(image)
            loss = criterion(out, label)
            _, pred = out.max(1)
            n_correct = pred.eq(label).float().mean()

            # visualize
            eval_loss += loss.item()
            eval_acc += n_correct.item()
            info['avg_eval_loss'] = '%.3f' % (eval_loss / (i + 1))
            info['avg_eval_acc'] = '%.3f' % (eval_acc / (i + 1) * 100)
            p_bar.set_postfix(info)

        p_bar.close()

        # save checkpoint
        if args.save_checkpoint:
            save_checkpoint(model, args)
    pass
