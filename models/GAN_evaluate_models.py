import GAN_models
import numpy as np
path = []
path.append('saved_model/20190603-232225/encoderA_epoch_100_weights.hdf5')
path.append('saved_model/20190603-232225/encoderB_epoch_100_weights.hdf5')
path.append('saved_model/20190603-232225/encoderShared_epoch_100_weights.hdf5')
path.append('saved_model/20190603-232225/decoderShared_epoch_100_weights.hdf5')
path.append('saved_model/20190603-232225/generatorA_epoch_100_weights.hdf5')
path.append('saved_model/20190603-232225/generatorB_epoch_100_weights.hdf5')

UNIT_GAN = GAN_models.UNIT()
ground_truths, outputs = UNIT_GAN.evaluate(path)
ground_truths = np.array(ground_truths)
outputs = np.array(outputs)
np.save('results/UNIT/ground_truth.npy', ground_truths)
np.save('results/UNIT/outputs.npy', outputs)