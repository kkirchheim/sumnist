from pytorch_ood.api import Detector
from pytorch_ood.utils import is_known
from torch import Tensor 
from typing import List, Dict 

class HybridMemory(Detector):
    """
    """
    def __init__(model, threshold=0.8):
        self.model = model 
        self.t = threshold
        
    def fit(dataset):
        """
        Labels key must be in 
        """
        ps = []
        combinations = set()
        for x, y in dataset: 
            l = [i.item() for i in y["labels"]]
            l.sort()
            combinations.add(tuple(l))
        
        return self 
    
    def fit_features(x): 
        raise NotImplementedError()
    
    def fit_features(x: List[Dict]):
        """
        Fit on the predictions 
        """
        
    def predict_features(x: List[Dict]) -> Tensor:
        """
        Dict must contains "labels" and "scores" keys. 
        """
        ood_scores = []
        scores = [s["scores"] for s in p]
        labels = [s["labels"] for s in p]
        
        for s, l in zip(scores, labels):
            v = [i.item() for i in l[s > self.t]]
            v.sort()

            if tuple(v) in combinations:
                ood_scores.append(0)
            else:
                ood_scores.append(1)
                
        return Tensor(scores) 
        
    def predict(x: Tensor) -> Tensor:
        with torch.no_grad():
            p = model(x)
        return self.predict_features(p)
    
      
    
class HybridSum(Detector):
    """
    """
    pass 
