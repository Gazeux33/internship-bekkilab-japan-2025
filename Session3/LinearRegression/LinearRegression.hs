module Main (main) where

import Torch.Tensor                        
import Torch.Functional (add, sub)
import qualified Torch.Functional.Internal as FI
import Torch.DType (DType(..))             
import Control.Monad (foldM)

ys :: [Float]
ys = [130, 195, 218, 166, 163, 155, 204, 270, 205, 127, 260, 249, 251, 158, 167]

xs :: [Float]
xs = [148, 186, 279, 179, 216, 127, 152, 196, 126, 78, 211, 259, 255, 115, 173]

learningRate :: Float
learningRate = 0.000001
epochs :: Int
epochs = 1

initialA, initialB :: Tensor
initialA = asTensor (0.0 :: Float)
initialB = asTensor (0.0 :: Float)

-- linear model: y = A*x + B
linear :: (Tensor, Tensor) -> Tensor -> Tensor
linear (slope, intercept) x = add (slope * x) intercept

main :: IO ()
main = do
  putStrLn $ "xs: " ++ show xs
  putStrLn $ "ys: " ++ show ys
  let lr = asTensor learningRate

  (finalA, finalB) <- foldM
    (\(a, b) epoch -> do
       putStrLn $ "\n *** Epoch " ++ show epoch ++ "/" ++ show epochs ++" ***"
       (newA, newB) <- foldM
         (\(currentA, currentB) (x, y) -> do
           let x_tensor = asTensor x
           let y_tensor = asTensor y
           let prediction = linear (currentA, currentB) x_tensor
           let loss = cost y_tensor prediction
           let newA = computeNewA x_tensor y_tensor currentA currentB learningRate
           let newB = computeNewB x_tensor y_tensor currentA currentB learningRate
           putStrLn $ "  loss=" ++ show loss
           putStrLn $ "  newA=" ++ show newA ++ ", newB=" ++ show newB
           putStrLn "------"
           pure (newA, newB)
         )
         (a, b)
         (zip xs ys)
       pure (newA, newB)
    )
    (initialA, initialB)
    [1..epochs]

  putStrLn $ "\nTraining finished. Final A=" ++ show finalA ++ ", B=" ++ show finalB

cost :: Tensor -> Tensor -> Tensor
cost actual predicted = 
    let diff = sub predicted actual
        squared = diff * diff
    in FI.meanAll squared Float

computeNewA :: Tensor -> Tensor -> Tensor -> Tensor -> Float -> Tensor
computeNewA x y a b lr =
    let pred = linear (a, b) x
        diff = sub pred y
        grad = FI.meanAll (2 * diff * x) Float  
    in  a - asTensor lr * grad

computeNewB :: Tensor -> Tensor -> Tensor -> Tensor -> Float -> Tensor
computeNewB x y a b lr =
    let pred = linear (a, b) x
        diff = sub pred y
        grad = FI.meanAll (2 * diff) Float      
    in  b - asTensor lr * grad