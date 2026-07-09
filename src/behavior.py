import scipy.io as sio
import numpy as np 
from utils.extract_vals import extract_value

class Behavior:

    def __init__(self, behaviorFile):

        ''' 
        args: the full behav.mat filepath
        '''
        behaviorFile = sio.loadmat(behaviorFile, struct_as_record=True)
        self.head = behaviorFile["Head"]
        self.reye = behaviorFile["Reye"] 
        self.sky = behaviorFile["Sky"]
        self.pupilDiameter = extract_value(self.reye["PupilDiameter"], 'PupilDiameter') ## this is too hardcoded. will change it later 
        
    def get_head_data(self):
        return self.head
    
    def get_reye_data(self):
        return self.reye
    
    def get_sky_data(self):
        return self.sky
    
    def get_pupilDiameter_data(self):
        return self.pupilDiameter