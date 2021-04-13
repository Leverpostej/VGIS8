import tensorflow as tf

def cyclic_loss(real, cycle):
    return tf.reduce_mean(tf.abs(real - cycle))

def mse_loss(real, fake):
    return tf.losses.mean_squared_error(real, fake)

def mae_loss(real, fake):
    return tf.reduce_mean(tf.losses.absolute_difference(real, fake))

def per_loss(real, fake):
    return tf.reduce_mean(tf.squared_difference(real, fake))

##################################################################################
# Loss function
##################################################################################
def gradient_loss(real, fake):   
    fake_gra = tf.image.image_gradients(fake)
    real_gra = tf.image.image_gradients(real)
    gra = tf.reduce_mean(tf.squared_difference(fake_gra, real_gra))
    return gra
    
    
def discriminator_loss(type, real, fake):
    real_loss = 0
    fake_loss = 0

    if type == 'lsgan' :
        real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0)) # b = 1
        fake_loss = tf.reduce_mean(tf.squared_difference(fake -1.0)) # a = -1

    if type == 'gan' :
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if type == 'wgan':
        real_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
        fake_loss = 0

    loss = real_loss + fake_loss

    return loss


def generator_loss(type, fake):
    fake_loss = 0

    if type == 'lsgan' :
        fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0)) # c = 1

    if type == 'gan' :
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if type == 'wgan':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss