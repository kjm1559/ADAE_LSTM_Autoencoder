import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

# model define
def encoder(input_dim, sequence_len, hidden_size, name='encoder'):
    '''
    Define lstm encoder
    Arg:
        input_dim (int): input dimension
        sequence_len (int): time sequence length
        hidden_size (int): hidden size
    Returns:
        model (keras.Model): keras model
    '''
    inputs = keras.Input(shape=(sequence_len, input_dim))
    # x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))(inputs)
    # x = keras.layers.TimeDistributed(keras.layers.Dropout(0.1))(x)
    
    # recurrent layers
    x = keras.layers.LSTM(hidden_size, return_sequences=True)(inputs)
    latent = keras.layers.LSTM(hidden_size)(x)
    return keras.Model(inputs, latent, name=name)

def decoder(output_dim, sequence_len, hidden_size, name='decoder'):
    '''
    Define lstm decoder
    Arg:
        output (int): output dimension
        sequence_len (int): time sequence length
        hidden_size (int): hidden size
    Returns:
        model (keras.Model): keras model
    '''
    inputs = keras.Input(shape=(hidden_size,))
    x = keras.layers.RepeatVector(sequence_len)(inputs)
    x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    x = keras.layers.LSTM(hidden_size, return_sequences=True)(x)
    
    x = keras.layers.TimeDistributed(keras.layers.Dense(512, activation='relu'))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x)
    
    x = keras.layers.TimeDistributed(keras.layers.Dense(128, activation='relu'))(x)
#     x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(axis=-1))(x)
#     x = keras.layers.TimeDistributed(keras.layers.Activation(keras.activations.relu))(x)
    x = keras.layers.TimeDistributed(keras.layers.Dropout(0.2))(x)
        
    x = keras.layers.TimeDistributed(keras.layers.Dense(32, activation='relu'))(x)
    
    recon = keras.layers.TimeDistributed(keras.layers.Dense(output_dim, activation='linear'))(x)
    return keras.Model(inputs, recon, name=name)

def autoencoder(input_dim=2, output_dim=2, sequence_len = 24, hidden_size=256):
    '''
    Define lstm autoencoder
    Arg:
        input_dim (int): input dimension
        output_dim (int): output dimension
        sequence_len (int): time sequence lenght
        hidden_size (int): hidden size
    Returns:
        model (keras.Model): keras model
    '''
    inputs = keras.Input(shape=(sequence_len, input_dim))
    # create model
    encoder_model = encoder(input_dim, sequence_len, hidden_size)
    decoder_model = decoder(output_dim, sequence_len, hidden_size)
    
    latent = encoder_model(inputs)
#     decoder_input = keras.layers.RepeatVector(sequence_len)(latent)
    outputs = decoder_model(latent)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='autoencoder')
    
    recon_loss = K.mean(keras.losses.mse(inputs[:, ::-1, :], outputs), axis=-1)
#     recon_loss = K.mean(keras.losses.mse(inputs, outputs), axis=-1)
    model.add_loss(recon_loss)    
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3))#, metric='mse')
    return model, encoder_model, decoder_model

# 1D sequence ADAE
def ADAE_LSTM(input_dim=2, output_dim=2, sequence_len = 24, hidden_size=256):
    g_encoder = encoder(input_dim, sequence_len, hidden_size, 'g_encoder')
    g_decoder = decoder(output_dim, sequence_len, hidden_size, 'g_decoder')

    d_encoder = encoder(input_dim, sequence_len, hidden_size, 'd_encoder')
    d_decoder = decoder(output_dim, sequence_len, hidden_size, 'd_decoder')

    inputs = keras.Input(shape=(sequence_len, input_dim))

    g_latent = g_encoder(inputs)
    g_outputs = g_decoder(g_latent)

    d_latent = d_encoder(inputs)
    d_outputs = d_decoder(d_latent)

    generator = keras.Model(inputs=inputs, outputs=g_outputs, name='generator_ae')
    discriminator = keras.Model(inputs=inputs, outputs=d_outputs, name='discriminator')
    
    # adae_lstm = keras.Model(inputs=inputs, outputs=d_outputs, name='adae_lstm')

    return generator, discriminator#, adae_lstm


class AdaeLstm(keras.Model):
    def __init__(self, generator, discriminator):
        super(AdaeLstm, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
    
    # def compile(self, gen_optimizer, dis_optimizer, **args):#optimizer=None, loss=None, metrics=None, weighted_metrics=None, loss_weights=None):
    def compile(self, optimizer=None, **args):#optimizer=None, loss=None, metrics=None, weighted_metrics=None, loss_weights=None):
        super(AdaeLstm, self).compile()
        self.gen_optimizer = optimizer[0]
        self.dis_optimizer = optimizer[0]
        
        
    def train_step(self, batch_data):
        with tf.GradientTape() as tape:
            x_gen = self.generator(batch_data, training=True)[:, ::-1, :]
            g_dis = self.discriminator(x_gen, training=True)[:, ::-1, :]
            
            x_gen = K.clip(x_gen, 0, 1)
            g_dis = K.clip(g_dis, 0, 1)
                
            loss_g = K.mean(K.square(batch_data - x_gen) + 0.5 * K.square(x_gen - g_dis))
            # loss_d = K.mean(K.abs(batch_data - x_dis) - K.abs(x_gen - g_dis))
            
        grads_G = tape.gradient(loss_g, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(grads_G, self.generator.trainable_variables))
        
        with tf.GradientTape() as tape:
            x_gen = self.generator(batch_data, training=True)[:, ::-1, :]
            x_dis = self.discriminator(batch_data, training=True)[:, ::-1, :]
            g_dis = self.discriminator(x_gen, training=True)[:, ::-1, :]

            x_gen = K.clip(x_gen, 0, 1)
            g_dis = K.clip(g_dis, 0, 1)
            x_dis = K.clip(x_dis, 0, 1)
              
            loss_d = K.mean(K.square(batch_data - x_dis) - 0.5 * K.square(x_gen - g_dis))

        grads_D = tape.gradient(loss_d, self.discriminator.trainable_variables)    
        self.dis_optimizer.apply_gradients(zip(grads_D, self.discriminator.trainable_variables))

        return {
            'gen_loss': loss_g,
            'dis_loss': loss_d,
            'gen_lr': self.gen_optimizer.lr,
            'dis_lr': self.dis_optimizer.lr
        } 

    def test_step(self, batch_data):
        x_gen = self.generator(batch_data, training=True)[:, ::-1, :]
        x_dis = self.discriminator(batch_data, training=True)[:, ::-1, :]
        g_dis = self.discriminator(x_gen, training=True)[:, ::-1, :]
        
        x_gen = K.clip(x_gen, 0, 1)
        g_dis = K.clip(g_dis, 0, 1)
        x_dis = K.clip(x_dis, 0, 1)
                
        loss_g = K.mean(K.square(batch_data - x_gen) + 0.5 * K.square(x_gen - g_dis))
        loss_d = K.mean(K.square(batch_data - x_dis) - 0.5 * K.square(x_gen - g_dis))

        return {
            'gen_loss': loss_g,
            'dis_loss': loss_d
        }  



