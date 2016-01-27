from blessings import Terminal

term = Terminal()

class Writer(object):

    """
    ---------------------------------------------------------------------------
    Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    ---------------------------------------------------------------------------
    """

    def __init__(self, location):

        """
        -----------------------------------------------------------------------
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        -----------------------------------------------------------------------
        """
        self.location = location

    def write(self, string):
        with term.location(*self.location):
            print(string)
        
