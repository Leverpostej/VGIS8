from Code.gan import GAN
from pip._internal.operations.freeze import freeze

def main():
    print([package for package in freeze()])
    restore_epoch = 50
    con_train = False
    w = 320
    h = 320
    c = 1
    
    lr = 0.0001
    m_s = 100
    b_s = 8
    
    direc = 'GAN-BC'

    test_gan = GAN(w, h, c, lr, m_s, b_s, restore_epoch, con_train, direc)
    test_gan.test()
    
if __name__ == '__main__':
    main()

