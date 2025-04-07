import numpy
import scipy.io
import datetime
ops = numpy.load('ops.npy', allow_pickle=True)
ops = ops.item()
ops["date_proc"] = ops["date_proc"].strftime('%Y-%m-%d_%H:%M:%S')
scipy.io.savemat('ops.mat', mdict={'ops': ops})
