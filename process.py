import tensorflow as tf
tf.enable_eager_execution()
import model as m
m.batchSize = 1
from model import model

import numpy as np
import scipy.io.wavfile as wavfile

model.load_weights('model.h5')

def split(left, right):
	def toFreq(samples):
		x = np.fft.fft(samples)
		il = np.swapaxes(np.asarray([np.real(x), np.imag(x)]), 0, 1)
		il = np.concatenate(il, axis=0) / 512
		return il

	input = np.concatenate(np.asarray([toFreq(left), toFreq(right)]))
	prediction = model.predict(np.asarray([input]))[0]
	buckets = [prediction[i * 512 * 2:i * 512 * 2 + 512 * 2] for i in xrange(9)]
	for i in xrange(9):
		bd = buckets[i] * 512
		af, bf = bd[::2], bd[1::2]
		il = np.asarray([np.complex(af[j], bf[j]) for j in xrange(512)])
		buckets[i] = np.real(np.fft.ifft(il))
	return buckets

def main(fn):
	fdata = wavfile.read(fn)[1] / 32768.
	fdata = np.swapaxes(fdata, 0, 1)
	buckets = [np.array([], dtype=np.int16) for i in xrange(9)]
	try:
		for i in xrange(0, len(fdata[0]), 512):
			if (i & 262143) == 0:
				print '%i/%i' % (i, len(fdata[0]))
			if i + 512 > len(fdata[0]):
				break
			tbuckets = split(fdata[0][i:i+512], fdata[1][i:i+512])
			for i in xrange(9):
				buckets[i] = np.concatenate([buckets[i], tbuckets[i]])
	except KeyboardInterrupt:
		pass
	
	for i in xrange(9):
		wavfile.write('bucket%i.wav' % i, 44100, np.asarray(buckets[i] * 24576, dtype=np.int16))

if __name__=='__main__':
	import sys
	main(*sys.argv[1:])
