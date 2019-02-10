import math, random, wave
import numpy as np
import scipy.io.wavfile

def noteToFreq(halfSteps):
	return 440 * (2 ** ((halfSteps - 49) / 12.))

sr = 44100
def sineOf(freq):
	x = np.arange(512)
	return np.sin(freq * np.pi * x / sr + freq / 2 * np.pi / sr)

notes = map(noteToFreq, range(30, 70))
noteSamples = map(sineOf, notes)

used = set()
for i in xrange(100):
	nc = random.randrange(1, 5)
	while True:
		cnotes = [random.randrange(len(notes)) for j in xrange(nc)]
		cnotes.sort()
		cnotes = tuple(cnotes)
		if cnotes not in used:
			used.add(cnotes)
			break

	wf = reduce(lambda a, x: a + x, [noteSamples[j] * random.uniform(0.25, 1) for j in cnotes]) / nc
	scipy.io.wavfile.write('sourceAudio/test%i.wav' % i, sr, np.asarray(wf * 24576, dtype=np.int16))
