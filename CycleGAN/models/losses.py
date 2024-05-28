import tensorflow as tf

adv_loss_fn = tf.keras.losses.BinaryCrossentropy()

class AdverserialLoss():
    # loss changed from binarycrossentropy(log) to mean squared error,as stated in section 4.2 

    @staticmethod
    def generator_loss(generated_img):
        # How far away is generated domain A to domainB
        loss_fn = tf.keras.losses.MeanSquaredError()
        loss = loss_fn(tf.ones_like(generated_img), generated_img)
        return loss

    @staticmethod
    def discriminator_loss(real, generated_img):

        # How far away is the generated image from the input img (domainA)
        # Disciminator needs to be able to properly classify real images as real (close to 1 [domainB])
        # Dsicriminator needs to classify fake/generated images as fake (close to 0 [domainA])
        loss_fn = tf.keras.losses.MeanSquaredError()
        real_loss = loss_fn(tf.ones_like(real), real) # maximise
        generated_loss = loss_fn(tf.zeros_like(generated_img),generated_img) # minimise
        
        total_loss = (real_loss + generated_loss) * 0.5
        return total_loss
   