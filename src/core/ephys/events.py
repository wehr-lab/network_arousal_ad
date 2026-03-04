from utils.util_funcs import extract_value, strcmp


class Events:
    """
    Initialize the Events class with fields as attributes.
    Args: one event from the eventsFile. Data type should be a numpy record array with named fields.
    Output: a class object with the fieldnames as attributes
    """

    def __init__(self, event):
        if not hasattr(event, "dtype") or not event.dtype.names:
            raise ValueError("Input should be a numpy record array with named fields.")

        self.fields = event.dtype.names

        ## set attributes for each field in the event

        for field in self.fields:
            value = extract_value(event[field])
            setattr(self, field, value)

    def __repr__(self):
        return f"Events({', '.join([f'{field}={getattr(self, field)}' for field in self.fields])})"
