from pytorch_ood.api import Detector
from pytorch_ood.utils import is_known
from torch import Tensor 
from typing import List, Dict 
import torch 


class HybridMemory(Detector):
    """
    Remember co-occurence of classes 
    """
    def __init__(self, model, threshold=0.8):
        self.model = model 
        self.t = threshold
        self.combinations = set()
        
    def fit(self, dataset):
        """
        Labels key must be in 
        """
        ps = []
        
        for x, y in dataset: 
            l = [i.item() for i in y["labels"]]
            l.sort()
            self.combinations.add(tuple(l))
        
        return self 
    
    def fit_features(self, x: List[Dict]): 
        """
        Fit on the predictions 
        """
        raise NotImplementedError()
        
    def predict_features(self, x: List[Dict]) -> Tensor:
        """
        Dict must contains "labels" and "scores" keys. 
        """
        if not self.combinations:
            raise ValueError()
            
        ood_scores = []
        scores = [s["scores"] for s in x]
        labels = [s["labels"] for s in x]
        
        for s, l in zip(scores, labels):
            v = [i.item() for i in l[s > self.t]]
            v.sort()

            if tuple(v) in self.combinations:
                ood_scores.append(0)
            else:
                ood_scores.append(1)
                
        return torch.tensor(ood_scores).float()
        
    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            p = self.model(x)
            
        return self.predict_features(p)
    
      
    
class HybridSum(Detector):
    """
    """
    def __init__(self, model, value=20, threshold=0.8):
        self.model = model 
        self.t = threshold
        self.value = value 
        self.combinations = set()
        
    def fit(self, dataset):
        return self 
    
    def fit_features(self, x: List[Dict]): 
        return self 
        
    def predict_features(self, x: List[Dict]) -> Tensor:
        """
        Dict must contains "labels" and "scores" keys. 
        """
        ood_scores = []
        scores = [s["scores"] for s in x]
        labels = [s["labels"] for s in x]
        
        for s, l in zip(scores, labels):
            v = [i.item() for i in l[s > self.t]]
            
            if sum(v) == self.value:
                ood_scores.append(0)
            else:
                ood_scores.append(1)
                
        return torch.tensor(ood_scores).float()
        
    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            p = self.model(x)
            
        return self.predict_features(p) 
