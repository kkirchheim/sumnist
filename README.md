# SuMNIST

This is the repository used to create the SuMNIST benchmark dataset. 
SuMNIST consists of images of size $56 \times 56$, each of which contains 4 instances from the MNIST dataset. 
For normal instances (that is, all instances in the training dataset) the numbers on the image sum to 20. 
On the test set, there are, however, anomalous instances where the numbers do not sum to 20, and that have to be detected. 

![](/img/mnist-example.png)


## Baselines

Baseline anomaly detection methods do not significantly outperform random guessing, even when using a pre-trained vision transformer as feature encoder. 
We also tested CLIP-based encoders.

|Method     |Backbone|AUROC|AUPR-IN|AUPR-OUT|FPR95TPR|
|-----------|--------|-----|-------|--------|--------|
|Deep SVDD  |-       |49.48|18.16  |81.63   |94.67   |
|1-NN       |-       |50.00|59.18  |90.82   |100.00  |
|Mahalanobis|-       |50.00|59.18  |90.82   |100.00  |
|Mahalanobis|ViT-L-16|50.00|59.18  |82.31   |100.00  |
|1-NN       |ViT-L-16|51.19|18.81  |82.21   |94.34   |



## Hybrid Approach 
With models based on a Mask-RCNN, we can easily solve the problem. 

![](/img/predictions.png)
