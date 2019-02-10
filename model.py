import tensorflow as tf
from tensorflow.keras import layers

batchSize = 125

def toTimeDomain(x):
	x *= 512
	rpart = tf.slice(x, [0, 0], [batchSize, 512 * 9])
	ipart = tf.slice(x, [0, 512 * 9], [batchSize, 512 * 9])
	complex = tf.complex(rpart, ipart)
	elements = []
	for i in xrange(9):
		td = tf.spectral.ifft(tf.slice(complex, [0, 512 * i], [batchSize, 512]))
		elements.append(tf.math.real(td))
	return tf.concat(elements, 1)

model = tf.keras.Sequential()
model.add(layers.Dense(2048, activation='tanh', input_dim=2048))
model.add(layers.Dense(1024, activation='tanh'))
model.add(layers.Dense(9216, activation='tanh'))
#model.add(layers.Lambda(toTimeDomain, output_shape=[batchSize, 512 * 9]))

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse', metrics=['mae'])
