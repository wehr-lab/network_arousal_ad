import pandas as pd
import numpy as np
from events import Events

class Session:

    def __init__(self, events=None):
        if events is None:
            self.events = []
        else:
            self.events = events

    def add_event(self, event):
        if isinstance(event, Events):
            self.events.append(event)
        else:
            raise ValueError('Input should be an instance of the Events class.')
    
    def get_event(self, index):
        if len(self.events) < np.abs(index): 
                raise ValueError('Index out of range.')
        else:
            return self.events[index]

    def __len__(self):
        return len(self.events)
    
    def toDataFrame(self):
        data = {field: [] for field in self.events[0].fields}
        for event in self.events:
            for field in event.fields:
                data[field].append(getattr(event, field))
        return pd.DataFrame(data)         
         

    def __repr__(self):
        return f'Session with {len(self.events)} events'
