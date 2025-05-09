module Main (main) where

import Torch
import qualified Torch.Functional.Internal as FI
import Control.Monad (foldM, when, forM_)

trainingData :: [([Float],Float)]
trainingData = [([1,1],1),([1,0],0),([0,1],0),([0,0],0)]

lr :: Tensor
lr = asTensor (0.1 :: Float)

epochs :: Int
epochs = 20

perceptron ::
    Tensor -> -- x
    Tensor -> -- w1
    Tensor -> -- w2
    Tensor -> -- bias
    Tensor    -- output
perceptron x w1 w2 b = FI.matmul (FI.cat [w1,w2] 0) x + b


myStep :: Tensor -> Tensor
myStep result = if (asValue result :: Float) > 0.0 then asTensor (1.0 :: Float) else asTensor (0.0 :: Float)


caluculateError ::
  Tensor -> -- y
  Tensor -> -- y'
  Tensor  -- error
caluculateError label input = label - input


--w_1 = w_1 + learning_rate * error[i] * x[i][0]
--w_2 = w_2 + learning_rate * error[i] * x[i][1]
--theta = theta + learning_rate*error[i]
trainStep ::
  Tensor -> -- x
  Tensor -> -- w1
  Tensor -> -- w2
  Tensor -> -- b
  Tensor -> -- y
  Tensor -> -- lr
  (Tensor,Tensor, Tensor, Tensor) -- (w1',w2', b', error)
trainStep x w1 w2 b y lr= (w1', w2', b', error)
  where
    output = myStep $ perceptron x w1 w2 b
    error = caluculateError y output
    w1' = w1 + (lr * error * (FI.select x 0 0))
    w2' = w2 + (lr * error * (FI.select x 0 1))
    b' = b + (lr * error)


train :: [([Float],Float)]-> -- data
  Tensor -> -- w1
  Tensor -> -- w2
  Tensor -> -- b
  Tensor -> -- lr
  Int -> -- epochs
  Int -> -- printFrequency
  IO (Tensor,Tensor, Tensor) -- (w1',w2', b')
train trainingData w1 w2 b lr epochs printFreq = do
    (finalW1,finalW2,finalB) <- foldM
        (\(w1, w2, b) epoch -> do
            let (newW1, newW2, newB, totalError) = foldl
                    (\(w1', w2', b', errSum) (x, y) -> 
                        let (w1'', w2'', b'', err) = trainStep (asTensor x) w1' w2' b' (asTensor y) lr
                        in (w1'', w2'', b'', errSum + Prelude.abs (asValue err :: Float)))
                    (w1, w2, b, 0.0)
                    trainingData
            
            if epoch `mod` printFreq == 0
                then putStrLn $ " *** Epoch " ++ show epoch ++ "/" ++ show epochs ++ " error=" ++ show totalError ++ " ***"
                else pure ()
            pure (newW1, newW2, newB)
        )
        (w1, w2, b)
        [1..epochs]
    return (finalW1, finalW2, finalB)
       



main :: IO ()
main = do
  w1 <- randIO' [1]
  w2 <- randIO' [1]
  b <- randIO' [1]

  putStrLn "Training perceptron for AND gate"
  (finalW1,finalW2,finalB) <- train trainingData w1 w2 b lr epochs 1
  putStrLn $ "Final weights: " ++ show finalW1 ++ ", " ++ show finalW2 ++ ", " ++ show finalB
  
  return ()