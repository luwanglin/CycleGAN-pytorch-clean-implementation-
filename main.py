import torch as t
from config import Config
from torchvision.transforms import transforms as tf
from PIL import Image
from dataset.dataset import ImageDataset
from torch.utils.data import DataLoader
from model import CycleGAN_model as CycleGan
import itertools
from utils.utils import LambdaLR, ReplayBuffer, Visualizer
import tqdm
from torchnet import meter
import time


def train(**kwargs):
    opt = Config()
    opt._parse(kwargs)

    transform = tf.Compose([
        tf.Resize(int(1.12 * opt.image_size), Image.BICUBIC),
        tf.RandomCrop(opt.image_size),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    '''
    Image.NEAREST ：低质量
    Image.BILINEAR：双线性
    Image.BICUBIC ：三次样条插值
    Image.ANTIALIAS：高质量
    '''
    # 读取数据
    trian_data = ImageDataset(opt.dataroot, transforms=transform, istrain=True)
    train_loader = DataLoader(trian_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    # 实例化网络
    G_A2B = CycleGan.generator()
    G_B2A = CycleGan.generator()

    D_A = CycleGan.discriminator()
    D_B = CycleGan.discriminator()

    if t.cuda.is_available():
        G_A2B.cuda()
        G_B2A.cuda()
        D_A.cuda()
        D_B.cuda()

    # 初始化网络
    G_A2B.weight_init()
    G_B2A.weight_init()
    D_A.weight_init()
    D_B.weight_init()

    # 定义loss
    criterion_GAN = t.nn.MSELoss()
    criterion_Cycle = t.nn.L1Loss()
    criterion_identity = t.nn.L1Loss()

    # 定义优化器
    optimizer_G = t.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()),
                               lr=opt.lr, betas=(opt.betas, 0.999))
    optimizer_D = t.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                               lr=opt.lr, betas=(opt.betas, 0.999))

    # 定义动态改变学习率
    lr_schedule_G = t.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                  lr_lambda=LambdaLR(opt.max_epoch, 0, opt.decay_epoch).step)
    lr_schedule_D = t.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                  lr_lambda=LambdaLR(opt.max_epoch, 0, opt.decay_epoch).step)

    # 输入输出,标签
    Tensor = t.cuda.FloatTensor if t.cuda.is_available() else t.Tensor
    input_A = Tensor(opt.batch_size, 3, opt.image_size, opt.image_size)
    input_B = Tensor(opt.batch_size, 3, opt.image_size, opt.image_size)
    target_real = t.ones(opt.batch_size, 1).cuda()
    target_fake = t.zeros(opt.batch_size, 1).cuda()

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 定义可视化visdom
    vis = Visualizer(env=opt.env, port=15024)

    # 定义averagemeter
    lossG_A2B_meter = meter.AverageValueMeter()
    lossG_B2A_meter = meter.AverageValueMeter()
    lossG_identity_meter = meter.AverageValueMeter()
    lossG_cycle_meter = meter.AverageValueMeter()
    lossD_B_meter = meter.AverageValueMeter()
    lossD_A_meter = meter.AverageValueMeter()

    # 开始训练
    lam = 10
    for epoch in range(opt.max_epoch):
        lossD_A_meter.reset()
        lossD_B_meter.reset()
        lossG_cycle_meter.reset()
        lossG_identity_meter.reset()
        lossG_B2A_meter.reset()
        lossG_A2B_meter.reset()
        for i, batch in tqdm.tqdm(enumerate(train_loader)):

            real_A = input_A.copy_(batch['A']).cuda()
            real_B = input_B.copy_(batch['B']).cuda()
            # print(real_A.requires_grad)
            # 训练生成器
            # 生成器A2b，生成器B2A
            optimizer_G.zero_grad()

            # identity loss
            # G_A2B(B)=B if B is real
            same_B = G_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 0.5 * lam
            # the same as above
            same_A = G_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 0.5 * lam
            lossG_identity_meter.add(loss_identity_A.item() + loss_identity_B.item())

            # GAN loss
            fake_B = G_A2B(real_A)
            prob_fakeB = D_B(fake_B)
            loss_GAN_A2B = criterion_GAN(prob_fakeB, target_real)
            lossG_A2B_meter.add(loss_GAN_A2B.item())

            fake_A = G_B2A(real_B)
            prob_fakeA = D_A(fake_A)
            loss_GAN_B2A = criterion_GAN(prob_fakeA, target_real)
            lossG_B2A_meter.add(loss_GAN_B2A.item())
            # Cycle loss
            recoverA = G_B2A(fake_B)
            loss_cycle_ABA = criterion_Cycle(recoverA, real_A) * lam

            recoverB = G_A2B(fake_A)
            loss_cycle_BAB = criterion_Cycle(recoverB, real_B) * lam
            lossG_cycle_meter.add(loss_cycle_BAB.item() + loss_cycle_ABA.item())
            # total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            optimizer_G.step()

            # 训练判别器
            optimizer_D.zero_grad()

            # real loss
            pred_real_B = D_B(real_B)
            loss_D_real_B = criterion_GAN(pred_real_B, target_real)

            # fake loss ,fake from buffer
            fake_B_new = fake_B_buffer.push_and_pop(fake_B)
            pred_fake_B = D_B(fake_B_new)
            loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)
            loss_total_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            lossD_B_meter.add(loss_total_B.item())
            loss_total_B.backward()

            # real loss
            pred_real_A = D_A(real_A)
            loss_D_real_A = criterion_GAN(pred_real_A, target_real)

            # fakr loss ,fake from buffer
            fake_A_new = fake_A_buffer.push_and_pop(fake_A)
            pred_fake_A = D_A(fake_A_new)
            loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake)
            loss_total_A = (loss_D_fake_A + loss_D_real_A) * 0.5
            lossD_A_meter.add(loss_total_A.item())
            loss_total_A.backward()

            optimizer_D.step()
            ###打印可视化
            if (i + 1) % opt.plot_every == 0:
                vis.plot('lossG_A2B', lossG_A2B_meter.value()[0])
                vis.plot('lossG_B2A', lossG_B2A_meter.value()[0])
                vis.plot('lossG_identity', lossG_identity_meter.value()[0])
                vis.plot('lossG_cycle', lossG_cycle_meter.value()[0])
                vis.plot('lossD_B', lossD_B_meter.value()[0])
                vis.plot('lossD_A', lossD_A_meter.value()[0])
                vis.img('real_A', real_A.data.cpu()[0] * 0.5 + 0.5)
                vis.img('fake_B', fake_B.data.cpu()[0] * 0.5 + 0.5)
                vis.img('real_B', real_B.data.cpu()[0] * 0.5 + 0.5)
                vis.img('fake_A', fake_A.data.cpu()[0] * 0.5 + 0.5)
        # 更新学习率
        lr_schedule_G.step()
        lr_schedule_D.step()

        # 保存模型m
        if (epoch + 1) % opt.savemode_every == 0:
            t.save(G_A2B.state_dict(), 'checkpoints/%s_%s_G_A2B.pth' % (epoch, time.strftime('%m%d_%H:%M%S')))
            t.save(G_B2A.state_dict(), 'checkpoints/%s_%s_G_B2A.pth' % (epoch, time.strftime('%m%d_%H:%M%S')))
            t.save(D_A.state_dict(), 'checkpoints/%s_%s_D_A.pth' % (epoch, time.strftime('%m%d_%H:%M%S')))
            t.save(D_B.state_dict(), 'checkpoints/%s_%s_D_B.pth' % (epoch, time.strftime('%m%d_%H:%M%S')))


'''
 在PyTorch 1.1.0之前的版本，学习率的调整应该被放在optimizer更新之前的。如果我们在 1.1.0 及之后的版本仍然将学习率的调整（即 scheduler.step()）放在 optimizer’s update（即 optimizer.step()）之前，那么 learning rate schedule 的第一个值将会被跳过。所以如果某个代码是在 1.1.0 之前的版本下开发，但现在移植到 1.1.0及之后的版本运行，发现效果变差，需要检查一下是否将scheduler.step()放在了optimizer.step()之前。
————————————————
版权声明：本文为CSDN博主「qyhaill」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qyhaill/article/details/103043637
'''

if __name__ == '__main__':
    train()
