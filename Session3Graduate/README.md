# Session3Graduate


## Launch the program
```bash
docker compose exec hasktorch /bin/bash -c "cd /home/ubuntu/new_repo/ && stack run Session3Graduate"
```

## Results 

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

