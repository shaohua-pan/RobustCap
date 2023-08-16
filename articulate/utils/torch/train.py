r"""
    Function to train a network.
"""


__all__ = ['train']


import os
import torch
from torch.utils.tensorboard import SummaryWriter
from thop import clever_format
import wandb

def train(net, train_dataloader, vald_dataloader=None, save_dir='weights', loss_fn=torch.nn.MSELoss(),
          eval_fn=None, optimizer=None, num_epoch=5000, num_iter_between_vald=-1, early_stop_threshold=-1,
          clip_grad_norm=0., load_last_states=True, save_log=True, eval_metric_names=None, epoch_callback_fn=None,
          wandb_project_name=None, wandb_config=None, wandb_watch=False, wandb_name=None, lr_scheduler_patience=None):
    r"""
    Train a net.

    Notes
    -----
    - The current weights, best weights, train info, optimizer states, and tensorboard logs
      (if `save_log` is True) will be saved into `save_dir` at the end of each validation.
    - When `vald_dataloader` is None, there is no validation and the best weight will be automatically updated
      whenever train loss decreases. Otherwise, it will be updated when validation loss decreases.
    - `*_dataloader` args are used as `for i, (data, label) in enumerate(dataloader)` and `len(dataloader)`.

    Args
    -----
    :param net: Network to train.
    :param train_dataloader: Train dataloader, enumerable and has __len__. It loads (train_data, train_label) pairs.
    :param vald_dataloader: Validation dataloader, enumerable and has __len__. It loads (vald_data, vald_label) pairs.
    :param save_dir: Directory for the saved model, weights, best weights, train information, etc.
    :param loss_fn: Loss function. Call like loss_fn(model(data), label). It should return one-element loss tensor.
    :param eval_fn: Eval function for validation. If None, use loss_fn for validation. It should return a tensor.
                    The very first element is used as the major validation loss (to save the weights).
    :param optimizer: Optimizer. If None, Adam is used by default and optimize net.parameters().
    :param num_epoch: Total number of training epochs. One epoch loads the entire dataset once.
    :param num_iter_between_vald: Number of training iterations between two consecutive validations.
                                  If negative, validations will be done once every epoch.
    :param early_stop_threshold: When vald loss does not decrease for early_stop_threshold times, training will stop
                                 early. If negative, the early stop strategy will not be applied.
    :param clip_grad_norm: 0 for no clip. Otherwise, clip the 2-norm of the gradient of net.parameters().
    :param load_last_states: If True and state files exist, last training states (net weights, optimizer states,
                             training info) will be loaded before training.
    :param save_log: If True, train-validation-loss curves will be plotted using tensorboard (saved in save_dir/log).
    :param eval_metric_names: A list of strings. Names of the returned values of eval_fn (used in tensorboard).
    :param epoch_callback_fn: If not None, call once at the end of each epoch.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(net.parameters())
    if eval_fn is None:
        eval_fn = loss_fn
    if eval_fn == loss_fn and eval_metric_names is None:
        eval_metric_names = ['loss']
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if wandb_project_name is None:
        wandb_project_name = 'default'
    if wandb_config is None:
        wandb_config = {}
    if wandb_name is None:
        wandb_name = 'default'
    wandb.init(project=wandb_project_name, name=wandb_name)
    if wandb_watch:
        wandb.watch(net, log='all', log_freq=100)
    if lr_scheduler_patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=lr_scheduler_patience, verbose=True)
    weights_file = os.path.join(save_dir, 'weights.pt')
    best_weights_file = os.path.join(save_dir, 'best_weights.pt')
    train_info_file = os.path.join(save_dir, 'train_info.pt')
    # structure_file = os.path.join(save_dir, 'structure.pt')
    optimizer_states_file = os.path.join(save_dir, 'optimizer_states.pt')

    min_vald_loss = 1e9
    num_iter_per_eopch = len(train_dataloader)
    num_train_step = min(num_iter_per_eopch, num_iter_between_vald) if num_iter_between_vald > 0 else num_iter_per_eopch
    num_vald_step = len(vald_dataloader) if vald_dataloader is not None else 0
    esn = early_stop_threshold if early_stop_threshold > 0 else float('inf')
    train_info = {'epoch': 0, 'it': 0, 'total_it': 0}
    writter = SummaryWriter(os.path.join(save_dir, 'log')) if save_log else None

    if load_last_states:
        if os.path.exists(train_info_file):
            train_info = torch.load(train_info_file)
            print('load train info, epoch: %d, iteration: %d, total_iteration: %s' %
                  (train_info['epoch'], train_info['it'], clever_format(train_info['total_it'], '%6.2f')))

        if os.path.exists(optimizer_states_file):
            optimizer.load_state_dict(torch.load(optimizer_states_file))
            print('load optimizer states')

        if os.path.exists(weights_file):
            net.load_state_dict(torch.load(weights_file))
            print('load weight file', end='')
            if vald_dataloader is not None:
                net.eval()
                with torch.no_grad():
                    vald_loss = sum([eval_fn(net(d), l) for d, l in vald_dataloader]) / num_vald_step
                    min_vald_loss = vald_loss.view(-1)[0].item()
                print(', vald_loss:', vald_loss.cpu(), ', min_val_loss:', min_vald_loss, end='')
            print('')

    # torch.save(net, structure_file)
    # print('the whole model (before training) is saved into structure.pt')
    total_it = train_info['total_it']

    for epoch in range(train_info['epoch'], num_epoch):
        net.train()
        train_loss = 0
        vald_loss_epoch = 0
        for i, (d, l) in enumerate(train_dataloader):
            if i < train_info['it']:
                continue
            loss = loss_fn(net(d), l)
            optimizer.zero_grad()
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_grad_norm)
            optimizer.step()
            train_loss += loss.item()
            total_it += 1

            if i % num_train_step == num_train_step - 1:  # validation
                net.eval()
                with torch.no_grad():
                    train_loss /= num_train_step
                    vald_loss = torch.tensor([train_loss]) if vald_dataloader is None else \
                        sum([eval_fn(net(d), l) for d, l in vald_dataloader]) / num_vald_step
                print('epoch: %-4d/%4d    iter: %-4d/%4d    total_it: %s    train_loss: %.6f    vald_loss: %s' %
                      (epoch, num_epoch, i + 1, num_iter_per_eopch, clever_format(total_it, '%6.2f'),
                       train_loss, vald_loss.cpu()), end='')
                wandb.log({'train_loss': train_loss, 'vald_loss': vald_loss.cpu().item()})
                vald_loss_epoch = vald_loss_epoch + vald_loss.cpu().item()
                torch.save(net.state_dict(), weights_file)
                torch.save(optimizer.state_dict(), optimizer_states_file)
                torch.save({'epoch': epoch, 'it': i + 1, 'total_it': total_it}, train_info_file)

                if save_log:
                    writter.add_scalar('train/loss', train_loss, total_it)
                    for idx, val in enumerate(vald_loss.view(-1)):
                        name = 'eval_fn[%d]' % idx if eval_metric_names is None else eval_metric_names[idx]
                        writter.add_scalar('valid/' + name, val, total_it)

                if vald_loss.view(-1)[0].item() < min_vald_loss:
                    min_vald_loss = vald_loss.view(-1)[0].item()
                    torch.save(net.state_dict(), best_weights_file)
                    esn = early_stop_threshold if early_stop_threshold > 0 else float('inf')
                    print('    best model is saved')
                else:
                    esn -= 1
                    print('    early stop' if esn == 0 else '')
                    if esn == 0:
                        return

                train_loss = 0
                net.train()
        if lr_scheduler_patience is not None:
            lr_scheduler.step(vald_loss_epoch)
        train_info['it'] = 0
        if epoch_callback_fn is not None:
            epoch_callback_fn()

    if save_log:
        writter.close()
