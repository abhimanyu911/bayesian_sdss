# bayesian_sdss
A Bayesian CNN approach to SDSS image data classfication

# Local installation

```
conda create -n env python=3.7

git clone https://github.com/abhimanyu911/bayesian_sdss.git

pip install -r requirements.txt
```

# Results 

| Model           | Parameters  | Training steps  | Accuracy    | F1-score    | AUC-ROC     |
| --------------- | ----------- | --------------- | ----------- | ----------- | ----------- |
| CNN(frequentist)| 149K        | 23850(50 epochs)| 0.80        | 0.79        | 0.9656      |
| CNN(Bayesian)   | 193K        | 11925(25 epochs)| 0.81        | 0.79        | 0.9671      |


The Bayesian CNN achieves marginally better performance in half the training steps. It is also more robust to overfitting as evidenced by the learning curves.


# Note

Kindly maintain tf-gpu/tf as 2.5.0 and tfp as 0.13.0 else you may encounter dependency issues
