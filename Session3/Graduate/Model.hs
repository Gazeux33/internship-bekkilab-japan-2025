module Model (
    trainModel,
    evalModel
) where

import Torch
import Control.Monad (foldM)
import qualified Torch.Functional.Internal as FI
import qualified Torch.Functional as F

learningRate :: Tensor
learningRate =  asTensor (0.000001 :: Float)
epochs :: Int
epochs = 1000

printFrequency :: Int
printFrequency = 100

initialA, initialB :: Tensor
initialA = asTensor (0.555 :: Float)
initialB = asTensor (94.585026 :: Float)

-- linear model: y = A*x + B
myLinear :: (Tensor, Tensor) -> Tensor -> Tensor
myLinear (slope, intercept) x = add (slope * x) intercept


trainModel :: [Float] -> [Float] -> [Float] -> [Float] -> IO (Tensor, Tensor, [Float])
trainModel xs_train ys_train xs_valid ys_valid = do
    let initialLosses = []
    (finalA, finalB, losses) <- foldM
        (\(a, b, losses) epoch -> do
           (newA, newB) <- foldM
             (\(currentA, currentB) (x, y) -> do
               let xTensor = asTensor x
                   yTensor = asTensor y
               let updatedA = computeNewA xTensor yTensor currentA currentB learningRate
                   updatedB = computeNewB xTensor yTensor currentA currentB learningRate
               pure (updatedA, updatedB)
             )
             (a, b)
             (zip xs_train ys_train)
           
           let currentLoss = evalModel xs_valid ys_valid (newA, newB)
           let lossValue = asValue currentLoss :: Float
           let newLosses = losses ++ [lossValue]
           
           if epoch `mod` printFrequency == 0
               then putStrLn $ " *** Epoch " ++ show epoch ++ "/" ++ show epochs ++" valid loss=" ++ show currentLoss ++ " ***"
               else pure ()
           
           pure (newA, newB, newLosses)
        )
        (initialA, initialB, initialLosses)
        [1..epochs]
    return (finalA, finalB, losses)





evalModel :: [Float] -> [Float] -> (Tensor, Tensor) -> Tensor
evalModel xs ys (a, b) = 
    let xsTensor = asTensor xs
        ysTensor = asTensor ys
        predictions = myLinear (a, b) xsTensor
        loss = cost ysTensor predictions
    in loss
 

cost :: Tensor -> Tensor -> Tensor
cost actual predicted = 
    let diff = sub predicted actual
        squared = diff * diff
    in FI.meanAll squared Float

computeNewA :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor
computeNewA x y a b lr =
    let pred = myLinear (a, b) x
        diff = sub pred y
        grad = FI.meanAll (2 * diff * x) Float  
    in  a - asTensor lr * grad

computeNewB :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor
computeNewB x y a b lr =
    let pred = myLinear (a, b) x
        diff = sub pred y
        grad = FI.meanAll (2 * diff) Float      
    in  b - asTensor lr * grad