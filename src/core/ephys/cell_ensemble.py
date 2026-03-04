import pandas as pd 
import numpy as np
from .cell import Cell

class CellEnsemble:

    '''Here cells is a list of Cell objects''' 
    def __init__(self, cells=None):
        if cells is None:
            self.cells = []
        else:
            self.cells = cells

    def add_cell(self, cell):
        if isinstance(cell, Cell):
            self.cells.append(cell)
        else:
            raise ValueError('Input should be a Cell object.')
        
    def get_cell(self, cellnum):
        for cell in self.cells:
            if cell.cellnum == cellnum + 1: ## had to to this because matlab starts at 1 and python at 0
                return cell
        return None
    
    def __len__(self):
        return len(self.cells)
    
    def __repr__(self):
        return f'CellEnsemble with {len(self)} cells'
    
    def toDataFrame(self):
        data = {field: [] for field in self.cells[0].fields}
        for cell in self.cells:
            for field in cell.fields:
                data[field].append(getattr(cell, field))
        return pd.DataFrame(data)         

