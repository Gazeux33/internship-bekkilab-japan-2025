module Main (main) where

import Torch
import qualified Torch.Functional.Internal as FI
import Torch.DType (DType(..))
import Control.Monad (when)   



xs :: Tensor
xs = FI.transpose (asTensor ([
    [93,230,250,260,119,183,151,192,263,185],
    [150,311,182,245,152,162,99,184,115,105]] :: [[Float]])) 0 1

ys :: Tensor
ys = asTensor ([123,290,230,261,140,173,133,179,210,181]:: [Float])

w :: Tensor
w = zeros' [2]

b :: Tensor
b = zeros' [1]

learningRate :: Tensor
learningRate = 1e-6

epochs :: Int
epochs = 1000


main :: IO ()
main = do

    
    putStrLn "\nstart training..."
    (finalW, finalB) <- trainLoop 0 epochs learningRate (w, b) xs ys
    
    let finalPred = myLinear (finalW, finalB) xs
    let finalLoss = cost ys finalPred
    
    putStrLn $ "Final loss: " ++ show finalLoss
    putStrLn $ "Final w: " ++ show finalW
    putStrLn $ "Final b: " ++ show finalB


trainLoop :: Int -> Int -> Tensor -> (Tensor, Tensor) -> Tensor -> Tensor -> IO (Tensor, Tensor)
trainLoop epoch maxEpochs lr (w, b) x y = do
    let y_pred = myLinear (w, b) x
        loss = cost y y_pred
        (dw, db) = gradients x y y_pred
        w' = w - (lr * dw)
        b' = b - (lr * db)
    
    when (epoch `mod` 100 == 0) $ do
        putStrLn $ "Epoch " ++ show epoch ++ " | Cost: " ++ show loss ++ " | w: " ++ show w' ++ " | b: " ++ show b'
    
    if epoch >= maxEpochs - 1
        then return (w', b')
        else trainLoop (epoch + 1) maxEpochs lr (w', b') x y

myLinear :: (Tensor, Tensor) -> Tensor -> Tensor
myLinear (w, b) x = matmul x w + b

cost :: Tensor -> Tensor -> Tensor
cost y_true y_pred = FI.meanAll (FI.pow (y_true - y_pred) 2) Float




gradients :: Tensor -> Tensor -> Tensor -> (Tensor, Tensor)
gradients x y_true y_pred = 
    let n_int = (FI.size x 0) :: Int
        n = asTensor ([fromIntegral n_int] :: [Float]) 
        error = y_pred - y_true
        dw = (2.0 / n) * matmul (transpose2D x) error
        db = (2.0 / n) * sumAll error
    in (dw, db)

