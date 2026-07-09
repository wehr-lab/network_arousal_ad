import pandas as pd
from events import Events

class EventsProcess():

    def __init__(self, event):
        if not isinstance(event, Events):
            raise ValueError('Input should be an instance of the Events class.')
        self.event = event  ## here event is an instance of the Events class. so when I need to access the fields of the event, I can do so by calling self.event.fieldname
        ## need to manually set the fields attribute
        self.stimtype = None 
        self.stimparam = None 
        self.stimDescription = None 
        self.OEStartTime = None
        self.OEEndTime = None 
        self.freqStart = None
        self.freqEnd = None
        self.firstFrame = None 
        self.lastFrame = None
        self.frameRate = None
        self.spikeTimes = []
        self.pupilDiameter = None 
        self.frameNums = []
        self.frameTimes = []
        self.timeWindow = None 
        self.frameWindow = None 
