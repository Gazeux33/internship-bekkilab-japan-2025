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

```bash
Epoc:  100  --->  Loss: 2.444038
Epoc:  200  --->  Loss: 2.409053
Epoc:  300  --->  Loss: 2.373184
Epoc:  400  --->  Loss: 2.290671
Epoc:  500  --->  Loss: 2.129993
Epoc:  600  --->  Loss: 1.930902
Epoc:  700  --->  Loss: 1.744073
Epoc:  800  --->  Loss: 1.532837
Epoc:  900  --->  Loss: 0.889003
Epoc: 1000  --->  Loss: 0.377688
[1.0,1.0]: Tensor Float []  0.1515   
[1.0,0.0]: Tensor Float []  0.8228   
[0.0,1.0]: Tensor Float []  0.7926   
[0.0,0.0]: Tensor Float []  0.2508  
```

## XOR

```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/internship-bekkilab-japan-2025/ && stack run sesssion4-xor"
```

Only the forward 