# SuMNIST

This is the repository used to create the SuMNIST benchmark dataset. 
SuMNIST consists of images of size $56 \times 56$, each of which contains 4 instances from the MNIST dataset. 
For the 60,000 normal instances in the training dataset, the numbers on the image sum to 20. 
In the test set with 10,000 images, there are, however, 8500 normal and 1500 anomalous instances where the numbers do not sum to 20, and that have to be detected. 

![](/img/mnist-example.png)


## Baselines

Baseline anomaly detection methods do not significantly outperform random guessing, even when using a pre-trained vision transformer as feature encoder. 
We also tested CLIP-based encoders.

|Method     |Backbone|AUROC|AUPR-IN|AUPR-OUT|FPR95   |
|-----------|--------|-----|-------|--------|--------|
|Deep SVDD  |-       |49.48|18.16  |81.63   |94.67   |
|1-NN       |-       |50.00|59.18  |90.82   |100.00  |
|Mahalanobis|-       |50.00|59.18  |90.82   |100.00  |
|Mahalanobis|ViT-L-16|50.00|59.18  |82.31   |100.00  |
|1-NN       |ViT-L-16|51.19|18.81  |82.21   |94.34   |



## Hybrid Approach 
With a hybrid model based on a Mask-RCNN, we can easily solve the problem. 
We propose two simple baselines for this kind of model.
HybridMem saves a set of all combinations of numbers that were observed during training. 
HybridSum simply calculates the sum of the numbers in the image and checks if it is equal to 20. 

|Method   |AUROC|AUPR-IN|AUPR-OUT|FPR95   |
|---------|-----|-------|--------|--------|
|HybridMem|95.30|82.72  |99.29   |9.26    |
|HybridSum|98.41|92.69  |99.76   |2.98    |

![](/img/predictions.png)







