# SuMNIST

The SuMNIST dataset comprises images with a size of $56 \times 56$, each containing 4 numbers from the MNIST dataset. 
In the training dataset, there are 60,000 normal instances where the numbers in the image sum to 20. 
However, in the test set with 10,000 images, there are 8,500 normal instances and 1,500 anomalous instances for which the numbers do not sum to 20.
The challenge is to detect these anomalies. 

![examples](/img/mnist-example.png)


## Baseline Methods

We evaluated various baseline anomaly detection methods on the SuMNIST dataset, including using pre-trained vision transformers as feature encoders and CLIP-based encoders.
Surprisingly, the baseline methods did not significantly outperform random guessing.


|Method     |Backbone|AUROC|AUPR-IN|AUPR-OUT|FPR95   |
|-----------|--------|-----|-------|--------|--------|
|Deep SVDD  |-       |49.48|18.16  |81.63   |94.67   |
|1-NN       |-       |50.00|59.18  |90.82   |100.00  |
|Mahalanobis|-       |50.00|59.18  |90.82   |100.00  |
|Mahalanobis|ViT-L-16|50.00|59.18  |82.31   |100.00  |
|1-NN       |ViT-L-16|51.19|18.81  |82.21   |94.34   |



## Hybrid Approach 

To address the limitations of the baseline methods, we propose a hybrid model based on a Faster-RCNN. 
This hybrid model offers improved performance on anomaly detection for the SuMNIST dataset. We introduce two simple baselines for this type of model.

* HybridMem: This approach saves a set of all combinations of numbers observed during training. During testing, it checks if the observed numbers match any of the saved combinations.
* HybridSum: This approach calculates the sum of the numbers in the image and checks if it is equal to 20.

|Method   |AUROC|AUPR-IN|AUPR-OUT|FPR95   |
|---------|-----|-------|--------|--------|
|HybridMem|95.30|82.72  |99.29   |9.26    |
|HybridSum|98.41|92.69  |99.76   |2.98    |

![example-predictions](/img/predictions.png)







