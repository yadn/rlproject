import torch
import numpy as np
from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_test_loader, get_train_valid_loader
from matplotlib import pyplot as plt

def imshow(inp, title=None):   
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array((0.1307,))
    print (mean)
    std  = np.array((0.3081,))
    print (std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)   
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def main(config):
    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_valid_loader(
            config.data_dir, config.batch_size,
            config.random_seed, config.valid_size,
            config.shuffle, config.show_sample, **kwargs
        )
    else:
        data_loader = get_test_loader(
            config.data_dir, config.batch_size, **kwargs
        )

#    for data, target in data_loader:
#        print(data.size())   
#        print(target.size())  
#        break
#    
#    inputs, classes = next(iter(data_loader))
#    print(data_loader)
#    out = torchvision.utils.make_grid(inputs)
#    class_names = np.arange(0,10)
#    imshow(out, title=[class_names[x] for x in classes])
        
    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
