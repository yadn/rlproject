import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import shutil
import pickle

from tqdm import tqdm
from utils import AverageMeter
from model import RecurrentAttention
from tensorboard_logger import configure, log_value


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 10
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram_{}_{}x{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale
        )

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
        )
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # # initialize optimizer and scheduler
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=self.momentum,
        # )
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=self.lr_patience
        # )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=3e-4,
        )

    def reset(self):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        h_t = torch.zeros(self.batch_size, self.hidden_size)
        h_t = Variable(h_t).type(dtype)

        l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
        l_t = Variable(l_t).type(dtype)

        return h_t, l_t

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):

            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            # train for 1 epoch
            train_loss, train_acc = self.train_one_epoch(epoch)

            # evaluate on validation set
            valid_loss, valid_acc = self.validate(epoch)

            # # reduce lr if validation loss plateaus
            # self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} "
            msg2 = "- val loss: {:.3f} - val acc: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, valid_loss, valid_acc))

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best
            )

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        # rew = torch.linspace(1,0.1,self.num_glimpses,dtype=float)
        rew = torch.logspace(0,-1,self.num_glimpses,dtype=float)
        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()

                # save images
                imgs = []
                imgs.append(x[0:9])

                # extract the glimpses
                locs = []
                log_pi = []
                baselines = []
                log_probas_list = []
                for t in range(self.num_glimpses):
                    # forward pass through model
                    h_t, l_t, b_t, log_probas1, p = self.model(
                    x, l_t, h_t, last=True)
                    # store
                    locs.append(l_t[0:9])
                    baselines.append(b_t)
                    log_pi.append(p)
                    predicted1 = torch.max(log_probas1, 1)[1]
                    R1 = (predicted1.detach() == y).float()
                    # R1[R1==0] = -1
                    log_probas_list.append(R1)

                # last iteration
                log_probas = log_probas1
                R = log_probas_list
                R = torch.stack(R).transpose(1,0)
                
                rew = rew.type(R.type())
                R = R.mul(rew)
                R[R==0] = -1
                # print(R)
                # h_t, l_t, b_t, log_probas, p = self.model(
                #     x, l_t, h_t, last=True
                # )
                # log_pi.append(p)
                # baselines.append(b_t)
                # locs.append(l_t[0:9])

                # convert list to tensors and reshape
                baselines = torch.stack(baselines).transpose(1, 0)
                log_pi = torch.stack(log_pi).transpose(1, 0)
                 
                # calculate reward
                predicted = torch.max(log_probas, 1)[1]
                R = (predicted.detach() == y).float()
                R = R.unsqueeze(1).repeat(1, self.num_glimpses)

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas, y)
                loss_baseline = F.mse_loss(baselines, R)

                # compute reinforce loss
                # summed over timesteps and averaged across batch
                adjusted_reward = R - baselines.detach()
                loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
                loss_reinforce = torch.mean(loss_reinforce, dim=0)

                # sum up into a hybrid loss
                loss = loss_action + loss_baseline + loss_reinforce

                # compute accuracy
                correct = (predicted == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])

                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss.data.item(), acc.data.item()
                        )
                    )
                )
                pbar.update(self.batch_size)

                # dump the glimpses and locs
                if plot:
                    if self.use_gpu:
                        imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                        locs = [l.cpu().data.numpy() for l in locs]
                    else:
                        imgs = [g.data.numpy().squeeze() for g in imgs]
                        locs = [l.data.numpy() for l in locs]
                    pickle.dump(
                        imgs, open(
                            self.plot_dir + "g_{}.p".format(epoch+1),
                            "wb"
                        )
                    )
                    pickle.dump(
                        locs, open(
                            self.plot_dir + "l_{}.p".format(epoch+1),
                            "wb"
                        )
                    )

                # log to tensorboard
                if self.use_tensorboard:
                    iteration = epoch*len(self.train_loader) + i
                    log_value('train_loss', losses.avg, iteration)
                    log_value('train_acc', accs.avg, iteration)

            return losses.avg, accs.avg

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            log_pi = []
            baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

                # store
                baselines.append(b_t)
                log_pi.append(p)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True
            )
            log_pi.append(p)
            baselines.append(b_t)

            # convert list to tensors and reshape
            baselines = torch.stack(baselines).transpose(1, 0)
            log_pi = torch.stack(log_pi).transpose(1, 0)

            # average
            log_probas = log_probas.view(
                self.M, -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            baselines = baselines.contiguous().view(
                self.M, -1, baselines.shape[-1]
            )
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(
                self.M, -1, log_pi.shape[-1]
            )
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach() == y).float()
            R = R.unsqueeze(1).repeat(1, self.num_glimpses)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas, y)
            loss_baseline = F.mse_loss(baselines, R)

            # compute reinforce loss
            adjusted_reward = R - baselines.detach()
            loss_reinforce = torch.sum(-log_pi*adjusted_reward, dim=1)
            loss_reinforce = torch.mean(loss_reinforce, dim=0)

            # sum up into a hybrid loss
            loss = loss_action + loss_baseline + loss_reinforce

            # compute accuracy
            correct = (predicted == y).float()
            acc = 100 * (correct.sum() / len(y))

            # store
            losses.update(loss.data.item(), x.size()[0])
            accs.update(acc.data.item(), x.size()[0])

            # log to tensorboard
            if self.use_tensorboard:
                iteration = epoch*len(self.valid_loader) + i
                log_value('valid_loss', losses.avg, iteration)
                log_value('valid_acc', accs.avg, iteration)

        return losses.avg, accs.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x, volatile=True), Variable(y)

            # duplicate 10 times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True
            )

            log_probas = log_probas.view(
                self.M, -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct) / (self.num_test)
        error = 100 - perc
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                correct, self.num_test, perc, error)
        )

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )
