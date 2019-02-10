import tensorflow as tf
from tensorflow.keras import layers
from glob import glob

tf.enable_eager_execution()

print 'Loading'
inputs = []
outputs = []
with file('training.bin', 'rb') as fp:
	for i in xrange(100000):
		outputs.append(tf.io.decode_raw(fp.read(512 * 9 * 2 * 4), tf.float32))
		inputs.append(tf.io.decode_raw(fp.read(512 * 2 * 2 * 4), tf.float32))
print 'Done loading'

from model import model, batchSize
model.load_weights('model.h5')

dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
dataset = dataset.batch(batchSize)
dataset = dataset.repeat()
model.fit(dataset, epochs=10, steps_per_epoch=400)

model.save_weights('model.h5', save_format='h5')
