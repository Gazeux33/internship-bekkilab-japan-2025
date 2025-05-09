# Session 3


## Linear Regression
```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run session3-linear-regression"
```

### Calculations

#### Gradient of A : 
$$
\frac{\partial \text{MSE}}{\partial a} = \frac{2}{n} \sum_{i=1}^{n} (a x_i + b - y_i) \cdot x_i
$$

---
#### Gradient of B : 

$$
\frac{\partial \text{MSE}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (a x_i + b - y_i)
$$

---

#### MSE Loss : 

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (a x_i + b - y_i)^2
$$

### Log
```bash
theocastillo@Mac-de-Theosoerus new_repo % docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/new_repo/ && stack run Session3"
stack: Ticker: poll failed: Interrupted system call: Interrupted system call
stack: Ticker: poll failed: Interrupted system call: Interrupted system call
stack: Ticker: poll failed: Interrupted system call: Interrupted system call
stack: Ticker: poll failed: Interrupted system call: Interrupted system call
stack: Ticker: poll failed: Interrupted system call: Interrupted system call
xs :[148.0,186.0,279.0,179.0,216.0,127.0,152.0,196.0,126.0,78.0,211.0,259.0,255.0,115.0,173.0]
ys :[130.0,195.0,218.0,166.0,163.0,155.0,204.0,270.0,205.0,127.0,260.0,249.0,251.0,158.0,167.0]
correct answer: Tensor Float [1] [ 130.0000   ]
estimated value: Tensor Float [1] [ 176.7250   ]
loss: Tensor Float [1] [ 2183.2290   ]
New A: Tensor Float [1] [-0.1365   ]
New B: Tensor Float [1] [ 94.5804   ]
******
correct answer: Tensor Float [1] [ 195.0000   ]
estimated value: Tensor Float [1] [ 197.8150   ]
loss: Tensor Float [1] [ 7.9244   ]
New A: Tensor Float [1] [ 0.5026   ]
New B: Tensor Float [1] [ 94.5847   ]
******
correct answer: Tensor Float [1] [ 218.0000   ]
estimated value: Tensor Float [1] [ 249.4300   ]
loss: Tensor Float [1] [ 987.8464   ]
New A: Tensor Float [1] [-0.3219   ]
New B: Tensor Float [1] [ 94.5819   ]
******
correct answer: Tensor Float [1] [ 166.0000   ]
estimated value: Tensor Float [1] [ 193.9300   ]
loss: Tensor Float [1] [ 780.0862   ]
New A: Tensor Float [1] [ 5.5053e-2]
New B: Tensor Float [1] [ 94.5822   ]
******
correct answer: Tensor Float [1] [ 163.0000   ]
estimated value: Tensor Float [1] [ 214.4650   ]
loss: Tensor Float [1] [ 2648.6489   ]
New A: Tensor Float [1] [-0.5566   ]
New B: Tensor Float [1] [ 94.5799   ]
******
correct answer: Tensor Float [1] [ 155.0000   ]
estimated value: Tensor Float [1] [ 165.0700   ]
loss: Tensor Float [1] [ 101.4057   ]
New A: Tensor Float [1] [ 0.4271   ]
New B: Tensor Float [1] [ 94.5840   ]
******
correct answer: Tensor Float [1] [ 204.0000   ]
estimated value: Tensor Float [1] [ 178.9450   ]
loss: Tensor Float [1] [ 627.7511   ]
New A: Tensor Float [1] [ 0.9358   ]
New B: Tensor Float [1] [ 94.5875   ]
******
correct answer: Tensor Float [1] [ 270.0000   ]
estimated value: Tensor Float [1] [ 203.3650   ]
loss: Tensor Float [1] [ 4440.2207   ]
New A: Tensor Float [1] [ 1.8610   ]
New B: Tensor Float [1] [ 94.5917   ]
******
correct answer: Tensor Float [1] [ 205.0000   ]
estimated value: Tensor Float [1] [ 164.5150   ]
loss: Tensor Float [1] [ 1639.0328   ]
New A: Tensor Float [1] [ 1.0651   ]
New B: Tensor Float [1] [ 94.5891   ]
******
correct answer: Tensor Float [1] [ 127.0000   ]
estimated value: Tensor Float [1] [ 137.8750   ]
loss: Tensor Float [1] [ 118.2663   ]
New A: Tensor Float [1] [ 0.4702   ]
New B: Tensor Float [1] [ 94.5839   ]
******
correct answer: Tensor Float [1] [ 260.0000   ]
estimated value: Tensor Float [1] [ 211.6900   ]
loss: Tensor Float [1] [ 2333.8530   ]
New A: Tensor Float [1] [ 1.5743   ]
New B: Tensor Float [1] [ 94.5899   ]
******
correct answer: Tensor Float [1] [ 249.0000   ]
estimated value: Tensor Float [1] [ 238.3300   ]
loss: Tensor Float [1] [ 113.8485   ]
New A: Tensor Float [1] [ 0.8314   ]
New B: Tensor Float [1] [ 94.5861   ]
******
correct answer: Tensor Float [1] [ 251.0000   ]
estimated value: Tensor Float [1] [ 236.1100   ]
loss: Tensor Float [1] [ 221.7107   ]
New A: Tensor Float [1] [ 0.9347   ]
New B: Tensor Float [1] [ 94.5865   ]
******
correct answer: Tensor Float [1] [ 158.0000   ]
estimated value: Tensor Float [1] [ 158.4100   ]
loss: Tensor Float [1] [ 0.1681   ]
New A: Tensor Float [1] [ 0.5503   ]
New B: Tensor Float [1] [ 94.5850   ]
******
correct answer: Tensor Float [1] [ 167.0000   ]
estimated value: Tensor Float [1] [ 190.6000   ]
loss: Tensor Float [1] [ 556.9617   ]
New A: Tensor Float [1] [ 0.1467   ]
New B: Tensor Float [1] [ 94.5827   ]
******

```

### Final parameters

**Epochs** :  ```1```

**lr** : ```0.000001```

Y = ```0.5732``` * X + ```94.5859```
```bash
Training finished. Final A=Tensor Float []  0.5732   , B=Tensor Float []  94.5859   
```

## Linear Regression with multiple X
```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run session3-multiple-x"
```


**learningRate** = ```1e-6```
**epochs** = ```1000```

```bash
Prédictions initiales : [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Coût initial : 39637.0
Gradients initiaux : (array([-78996.8, -70601.8]), np.float64(-384.0))
```


```bash
start training...
Epoch 0 | Cost: Tensor Float []  39637.0000    | w: Tensor Float [2] [ 7.8997e-2,  7.0602e-2] | b: Tensor Float [1] [ 3.8400e-4]
Epoch 100 | Cost: Tensor Float []  120.4294    | w: Tensor Float [2] [ 0.5728   ,  0.4784   ] | b: Tensor Float [1] [ 2.7276e-3]
Epoch 200 | Cost: Tensor Float []  117.3162    | w: Tensor Float [2] [ 0.5844   ,  0.4655   ] | b: Tensor Float [1] [ 2.7451e-3]
Epoch 300 | Cost: Tensor Float []  116.0121    | w: Tensor Float [2] [ 0.5920   ,  0.4571   ] | b: Tensor Float [1] [ 2.7568e-3]
Epoch 400 | Cost: Tensor Float []  115.4657    | w: Tensor Float [2] [ 0.5969   ,  0.4516   ] | b: Tensor Float [1] [ 2.7648e-3]
Epoch 500 | Cost: Tensor Float []  115.2369    | w: Tensor Float [2] [ 0.6000   ,  0.4481   ] | b: Tensor Float [1] [ 2.7705e-3]
Epoch 600 | Cost: Tensor Float []  115.1410    | w: Tensor Float [2] [ 0.6021   ,  0.4458   ] | b: Tensor Float [1] [ 2.7745e-3]
Epoch 700 | Cost: Tensor Float []  115.1009    | w: Tensor Float [2] [ 0.6034   ,  0.4444   ] | b: Tensor Float [1] [ 2.7776e-3]
Epoch 800 | Cost: Tensor Float []  115.0841    | w: Tensor Float [2] [ 0.6043   ,  0.4434   ] | b: Tensor Float [1] [ 2.7800e-3]
Epoch 900 | Cost: Tensor Float []  115.0769    | w: Tensor Float [2] [ 0.6048   ,  0.4428   ] | b: Tensor Float [1] [ 2.7820e-3]
Final loss: Tensor Float []  115.0740   
Final w: Tensor Float [2] [ 0.6052   ,  0.4424   ]
Final b: Tensor Float [1] [ 2.7837e-3]
```

## Linear regression with CSV Data (Graduate)
```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run session3-graduate"
```
```X``` = GRE Score        ```Y``` = TOEFL Score

### Results 

```bash
Start...
x_train:320 y_train:320
x_eval:40 y_eval:40
x_valid:40 y_valid:320
Start train
 *** Epoch 100/1000 valid loss=Tensor Float []  28.6154    ***
 *** Epoch 200/1000 valid loss=Tensor Float []  28.6149    ***
 *** Epoch 300/1000 valid loss=Tensor Float []  28.6145    ***
 *** Epoch 400/1000 valid loss=Tensor Float []  28.6140    ***
 *** Epoch 500/1000 valid loss=Tensor Float []  28.6136    ***
 *** Epoch 600/1000 valid loss=Tensor Float []  28.6131    ***
 *** Epoch 700/1000 valid loss=Tensor Float []  28.6127    ***
 *** Epoch 800/1000 valid loss=Tensor Float []  28.6122    ***
 *** Epoch 900/1000 valid loss=Tensor Float []  28.6118    ***
 *** Epoch 1000/1000 valid loss=Tensor Float []  28.6114    ***
Final A: Tensor Float []  3.8934e-2
Final B: Tensor Float []  94.5681   
Final loss: Tensor Float []  38.9876   
End train 
Plot Result
Learning curve saved to outpout/train.png

```



## Estimate chances of admission 
```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run session3-admission"
```