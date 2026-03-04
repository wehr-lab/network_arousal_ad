import pandas as pd
import numpy as np
from .events_process import EventsProcess 

class SessionProcess:

    def __init__(self, eventsProcess=None):
        if eventsProcess is None:
            self.eventsProcess = []
        else:
            self.eventsProcess = eventsProcess

    def add_eventProcess(self, eventProcess):
        if isinstance(eventProcess, EventsProcess):
            self.eventsProcess.append(eventProcess)
        else:
            raise ValueError('Input should be an instance of the EventsProcess class.')

    def toDataFrame(self):
        # Check if eventsProcess list is empty
        if not self.eventsProcess:
            raise ValueError('No events to process.')
        
        # Get all unique attributes from all events (excluding private/special methods)
        all_attributes = set()
        for eventProcess in self.eventsProcess:
            # Get all attributes that don't start with underscore
            attributes = [attr for attr in dir(eventProcess) 
                         if not attr.startswith('_') and not callable(getattr(eventProcess, attr))]
            all_attributes.update(attributes)
        
        # Initialize data dictionary with all attributes
        data = {attr: [] for attr in all_attributes}
        
        # Populate data
        for eventProcess in self.eventsProcess:
            for attr in all_attributes:
                if hasattr(eventProcess, attr):
                    data[attr].append(getattr(eventProcess, attr))
                else:
                    data[attr].append(None)  # Fill missing attributes with None
        
        return pd.DataFrame(data)
