from abc import ABC, abstractmethod 

class DynamicalSystem(ABC):
    @abstractmethod 
    def initialize(self, config):
        @abstractmethod
        def initialize(self, config):
            """
            Initialize the dynamical system with the provided configuration.
            """
            pass

        @abstractmethod
        def derivative(self, state):
            """
            Compute the derivative of the system at the given state.
            """
            pass

        @abstractmethod
        def return_state(self):
            """
            Return the current state of the system.
            """
            pass
        