from utils.extract_vals import extract_value

class Cell:
    def __init__(self, sortedUnits):
        
        if not hasattr(sortedUnits, 'dtype') or not sortedUnits.dtype.names:
            raise ValueError('Input should be a numpy record array with named fields.')
        
        self.fields = sortedUnits.dtype.names

        ## set attributes for each field in the event

        for field in self.fields:
            value = extract_value(sortedUnits[field], field)
            setattr(self, field, value)

        def __repr__(self):
            return f'Cell {self.cellnum}'