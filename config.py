import warnings


class Config(object):
    max_epoch = 200
    batch_size = 1
    dataroot = '/data/ThreeTeamData/horse2zebra'
    lr = 0.0002
    decay_epoch = 100
    image_size = 256
    input_nc = 3
    GPU = True
    output_image = 'output/'
    plot_every = 10  # 多少个batch画损失

    savemode_every = 10  # 多少epoch保存一次模型

    netd_path = None  # 预训练模型
    netg_path = None
    num_workers = 4
    betas=0.5#Adam优化器中的第一个参数
    env='CycleGAN'#visdom环境

    def _parse(self, kwargs):
        '''
        根据字典进行更新配置
        :param kwargs:
        :return:
        '''
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
                exit()
            setattr(self, k, v)
        print('user config')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))
