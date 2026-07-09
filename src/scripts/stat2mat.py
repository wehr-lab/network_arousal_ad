import numpy, scipy.io
stat = numpy.load('stat.npy', allow_pickle=True)
scipy.io.savemat('stat.mat', mdict={'stat': stat})
