from Code.gan import GAN
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

def main():    
# new training
    restore_epoch = 0
    con_train = False
# continuing training from previous stop point
#    restore_epoch = 50
#    con_train = True
    w = 80
    h = 80
    c = 1
    
    lr = 0.0001
    m_s = 100
    b_s = 64
    
    direc = 'ZP-GAN'

    train_gan = GAN(w, h, c, lr, m_s, b_s, restore_epoch, con_train, direc)
    train_gan.train()

if __name__ == '__main__':
    main()

