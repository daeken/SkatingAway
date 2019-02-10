import random, struct
from glob import glob
import numpy as np
import scipy.io.wavfile as wavfile

allSources = []
for fn in glob('sourceAudio/*.wav'):
	fdata = wavfile.read(fn)[1] / 32768.
	if len(fdata) == 512:
		allSources.append(fdata)
		continue
	
	fdata = np.swapaxes(fdata, 0, 1)
	for channel in fdata:
		for i in xrange(0, len(channel), 512):
			if len(channel) - i >= 512:
				allSources.append(channel[i:i+512])

def inBucket(num, location):
	if location <= -0.9:
		return num == 0
	elif location >= 0.9:
		return num == 8

	bucketRange = 2 / 9.
	center = (num + .5) * bucketRange - 1
	hr = bucketRange / 2
	
	return location > center - hr and location <= center + hr

def buildOne():
	sources = random.sample(allSources, random.randrange(1, 4))
	locations = [random.uniform(-1, 1) for _ in sources]

	left = reduce(lambda a, x: a + x, [sources[i] * (max(-locations[i], 0) + (1 - max(locations[i], 0))) for i in xrange(len(sources))]) / len(sources)
	right = reduce(lambda a, x: a + x, [sources[i] * (max(locations[i], 0) + (1 - max(-locations[i], 0))) for i in xrange(len(sources))]) / len(sources)

	top = max(np.max(left), np.max(-left))
	if top == 0:
		top = 1
	left /= top
	top = max(np.max(right), np.max(-right))
	if top == 0:
		top = 1
	right /= top

	def buildFFTData(x):
		x = np.fft.fft(x)
		il = np.swapaxes(np.asarray([np.real(x), np.imag(x)]), 0, 1)
		il = np.concatenate(il, axis=0) / 512
		assert np.max(il) <= 1 and np.min(il) >= -1
		return struct.pack('<' + 'f' * len(il), *il)

	edata = ''
	for i in xrange(9):
		bucket = np.array([0.0] * 512)

		for j in xrange(len(sources)):
			if inBucket(i, locations[j]):
				bucket += sources[j]

		edata += buildFFTData(bucket)

	ldata = buildFFTData(left)
	rdata = buildFFTData(right)

	return edata + ldata + rdata

with file('training.bin', 'wb') as fp:
	for i in xrange(100000):
		if (i % 1000) == 0:
			print i
		fp.write(buildOne())
