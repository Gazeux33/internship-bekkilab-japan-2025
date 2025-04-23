# Session3


## Launch the program
```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/new_repo/ && stack run Session3"
```


## Calculations

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

## Log
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

## Final parameters

Y = ```0.5732``` * X + ```94.5859```
```bash
Training finished. Final A=Tensor Float []  0.5732   , B=Tensor Float []  94.5859   
```


