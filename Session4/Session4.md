# Session 4

## AND

```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run session4-and"
```

**lr** = ```0.1 ```

**epochs** =``` 20```

### Launch the programm 

```bash
Training perceptron for AND gate
 *** Epoch 1/20 error=3.0 ***
 *** Epoch 2/20 error=3.0 ***
 *** Epoch 3/20 error=3.0 ***
 *** Epoch 4/20 error=2.0 ***
 *** Epoch 5/20 error=1.0 ***
 *** Epoch 6/20 error=2.0 ***
 *** Epoch 7/20 error=0.0 ***
 *** Epoch 8/20 error=0.0 ***
 *** Epoch 9/20 error=0.0 ***
 *** Epoch 10/20 error=0.0 ***
 *** Epoch 11/20 error=0.0 ***
 *** Epoch 12/20 error=0.0 ***
 *** Epoch 13/20 error=0.0 ***
 *** Epoch 14/20 error=0.0 ***
 *** Epoch 15/20 error=0.0 ***
 *** Epoch 16/20 error=0.0 ***
 *** Epoch 17/20 error=0.0 ***
 *** Epoch 18/20 error=0.0 ***
 *** Epoch 19/20 error=0.0 ***
 *** Epoch 20/20 error=0.0 ***
Final weights: Tensor Float [1] [ 0.2132   ], Tensor Float [1] [ 0.1610   ], Tensor Float [1] [-0.3429   ]
```



## NAND

```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run sesssion4-nand"
```

**lr** = ```1e-1 ```

**epochs** =``` 1500```

### Tanh 
```bash
Epoc:  100  --->  Loss: 1.799714
Epoc:  200  --->  Loss: 0.152960
Epoc:  300  --->  Loss: 0.055934
Epoc:  400  --->  Loss: 0.032687
Epoc:  500  --->  Loss: 0.022751
Epoc:  600  --->  Loss: 0.017329
Epoc:  700  --->  Loss: 0.013941
Epoc:  800  --->  Loss: 0.011635
Epoc:  900  --->  Loss: 0.009968
Epoc: 1000  --->  Loss: 0.008709
Epoc: 1100  --->  Loss: 0.007727
Epoc: 1200  --->  Loss: 0.006939
Epoc: 1300  --->  Loss: 0.006294
Epoc: 1400  --->  Loss: 0.005757
Epoc: 1500  --->  Loss: 0.005303
[1.0,1.0]: Tensor Float []  2.3680e-2
[1.0,0.0]: Tensor Float []  0.9774   
[0.0,1.0]: Tensor Float []  0.9719   
[0.0,0.0]: Tensor Float []  1.5993e-2 
```

### Sigmoid
```bash
Epoc:  100  --->  Loss: 2.424500
Epoc:  200  --->  Loss: 2.402122
Epoc:  300  --->  Loss: 2.391043
Epoc:  400  --->  Loss: 2.372428
Epoc:  500  --->  Loss: 2.309883
Epoc:  600  --->  Loss: 2.110009
Epoc:  700  --->  Loss: 1.795820
Epoc:  800  --->  Loss: 1.526996
Epoc:  900  --->  Loss: 1.158576
Epoc: 1000  --->  Loss: 0.609225
Epoc: 1100  --->  Loss: 0.285158
Epoc: 1200  --->  Loss: 0.162313
Epoc: 1300  --->  Loss: 0.108045
Epoc: 1400  --->  Loss: 0.079228
Epoc: 1500  --->  Loss: 0.061824
[1.0,1.0]: Tensor Float []  6.8059e-2
[1.0,0.0]: Tensor Float []  0.9349   
[0.0,1.0]: Tensor Float []  0.9120   
[0.0,0.0]: Tensor Float []  9.9003e-2
```

### Relu

```bash
Epoc:  100  --->  Loss: 2.500000
Epoc:  200  --->  Loss: 2.500000
Epoc:  300  --->  Loss: 2.500000
Epoc:  400  --->  Loss: 2.500000
Epoc:  500  --->  Loss: 2.500000
Epoc:  600  --->  Loss: 2.500000
Epoc:  700  --->  Loss: 2.500000
Epoc:  800  --->  Loss: 2.500000
Epoc:  900  --->  Loss: 2.500000
Epoc: 1000  --->  Loss: 2.500000
Epoc: 1100  --->  Loss: 2.500000
Epoc: 1200  --->  Loss: 2.500000
Epoc: 1300  --->  Loss: 2.500000
Epoc: 1400  --->  Loss: 2.500000
Epoc: 1500  --->  Loss: 2.500000
[1.0,1.0]: Tensor Float []  0.5000   
[1.0,0.0]: Tensor Float []  0.5000   
[0.0,1.0]: Tensor Float []  0.5000   
[0.0,0.0]: Tensor Float []  0.5000 
```


## XOR

```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run sesssion4-xor"
```

Only the forward propagation