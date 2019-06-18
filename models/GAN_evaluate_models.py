import GAN_models
import numpy as np
'''
path = []
path.append('models/saved_models/20190603-232225/encoderA_epoch_100_weights.hdf5')
path.append('models/saved_models/20190603-232225/encoderB_epoch_100_weights.hdf5')
path.append('models/saved_models/20190603-232225/encoderShared_epoch_100_weights.hdf5')
path.append('models/saved_models/20190603-232225/decoderShared_epoch_100_weights.hdf5')
path.append('models/saved_models/20190603-232225/generatorA_epoch_100_weights.hdf5')
path.append('models/saved_models/20190603-232225/generatorB_epoch_100_weights.hdf5')

UNIT_GAN = GAN_models.UNIT()
inputs, ground_truths, outputs = UNIT_GAN.evaluate(path)
inputs = np.array(inputs)
ground_truths = np.array(ground_truths)
outputs = np.array(outputs)
np.save('models/results/UNIT/inputs.npy', inputs)
np.save('models/results/UNIT/ground_truth.npy', ground_truths)
np.save('models/results/UNIT/outputs.npy', outputs)
'''
path = []
path.append('models/saved_models/20190524-131743/G_A2B_model_weights_epoch_200.hdf5')
path.append('models/saved_models/20190524-131743/G_B2A_model_weights_epoch_200.hdf5')
path.append('models/saved_models/20190524-131743/D_A_model_weights_epoch_200.hdf5')
path.append('models/saved_models/20190524-131743/D_B_model_weights_epoch_200.hdf5')
CGAN = GAN_models.CycleGAN()
fakeAs, fakeBs = CGAN.evaluate(path)
#np.save('models/results/fakeAs.npy', fakeAs)
#np.save('models/results/fakeBs.npy', fakeBs)
#%%
#print(CGAN.G_model.metrics_names)

#2.993938446044922, 0.053988561034202576, 0.048053331673145294, 0.9999938011169434, 0.973525881767273
