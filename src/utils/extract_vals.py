import numpy as np 

def extract_value(arr, fieldname=None):
    '''Only run this function after importing numpy as np. 
    
    input: arr, a numpy array or list
    output: the very first value from the array or list i.e. arr[0][0]...
    '''
    if fieldname == 'spiketimes':
        if isinstance(arr, np.ndarray):
            return np.array(arr[0])
        elif isinstance(arr, list):
            return arr  # Return the list as is
        else:
            return None  # Handle unsupported type

    if fieldname == 'PupilDiameter':
        return arr[0][0][:, 0] ## I was lazy to write a proper function. This is a temporary solution 

    while True:
        ## check if the current item is a NumPy array or a list
        if isinstance(arr, (np.ndarray, list)):
            if len(arr) > 0:  # Ensure the array or list is not empty
                arr = arr[0]
            else:
                return None  # Return None for empty arrays or lists
        # Check if the value is a NumPy integer or float
        elif isinstance(arr, (np.integer, np.floating, np.str_)):
            return arr.item()  # Convert to plain Python type
        # Check if the value is a plain Python integer or float
        elif isinstance(arr, (int, float)):
            return arr
        else:
            return None  # Return None for unsupported types
