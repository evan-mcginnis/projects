#
# E X C E P T I O N S
#

class Exceptions(Exception):
    def __init__(self, message):
        self._message = message
        super().__init__(self._message)

    def __str__(self):
        return f'{self._message}'

class PersistenceError(Exceptions):
    def __init__(self, message):
        super().__init__(message)

class ProcessingError(Exceptions):
    def __init__(self, message):
        super().__init__(message)

class EOL(Exceptions):
    def __init___(self, message):
        super().__init__(message)