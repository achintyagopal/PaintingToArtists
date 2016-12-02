from abc import ABCMeta, abstractmethod

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._label = label
        
    def __str__(self):
        return str(self._label)

class Instance:

    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label

    def get_vector(self):
        return self._feature_vector

    def get_label(self):
        return self._label

    def __iter__(self): 
        return iter((self.feature_vector, self.label))

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, feature_converter): 
        pass

    @abstractmethod
    def predict(self, feature_converter): 
        pass

       
class FeatureConverter():
    __metaclass__ = ABCMeta

    @abstractmethod
    def createTrainingInstances(self, files):
        pass
    
    @abstractmethod
    def createTestingInstances(self, files):
        pass
    
    @abstractmethod
    def getTrainingInstance(self, index):
        pass
    
    @abstractmethod
    def getTestingInstance(self, index):
        pass
    
    @abstractmethod
    def trainingInstancesSize(self):
        pass
    
    @abstractmethod
    def testingInstanceSize(self):
        pass

