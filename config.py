# global config
class Config:
    # NN model
    num_filters = 64
    kernel_size = 3
    hidden_dim = 64
    flow_hidden_dim = 64

    # training
    lr = 0.0001  # optimizer has learning rate {}
    entropy_lr = 0.1  # entropy item in gradient has {} weight

    # PPO
    k_epoch = 10
    eps_clip = 0.2
    # ppo_lr = 1e-4  # fan
    ppo_lr = 0.001  # sns
    ppo_entropy_lr = 0.2

    # PPOGCN
    hidden_features = 128

    # other
    log_level = 'INFO'
    seed = 1024  # random seed used for splitting dataset

    def __repr__(self):
        vs = list(filter(lambda x: '__' not in x[0], vars(Config).items()))
        return '\n'.join(f'{k}: {v}' for k, v in vs)


config = Config()

if __name__ == '__main__':
    print(config)
