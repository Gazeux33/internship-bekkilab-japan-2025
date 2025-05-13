module Evaluation (
    accuracy,
    binaryConfusionMatrix,
    precision,
    recall,
    f1Score,
    roundTensor
    , multiclassConfusionMatrix
    , printConfusionMatrix
)where 

import Torch hiding (div)
import qualified Torch.Functional.Internal as FI
import Control.Monad (forM_)
import Text.Printf     (printf)



roundTensor :: Tensor -> Tensor
roundTensor t = 
    let rounded = FI.round_t t
    in rounded


accuracy :: Tensor -> Tensor -> Float
accuracy yTrue yPred = 
    let correctPred = asValue (sumAll (eq yTrue yPred)) :: Int
        nbPred = FI.size yPred 0
    in fromIntegral correctPred / fromIntegral nbPred


binaryConfusionMatrix :: Tensor -> Tensor -> (Int, Int, Int, Int)
binaryConfusionMatrix yTrue yPred =
    let tp = asValue (sumAll (logicalAnd (eq yTrue 1) (eq yPred 1))) :: Int
        tn = asValue (sumAll (logicalAnd (eq yTrue 0) (eq yPred 0))) :: Int
        fp = asValue (sumAll (logicalAnd (eq yTrue 0) (eq yPred 1))) :: Int
        fn = asValue (sumAll (logicalAnd (eq yTrue 1) (eq yPred 0))) :: Int
    in (tp, fp, tn, fn)



multiclassConfusionMatrix :: Tensor -> Tensor -> [[Int]]
multiclassConfusionMatrix yTrue yPred =
    let yTrueList = map round (asValue yTrue :: [Float])
        yPredList = map round (asValue yPred :: [Float])
        pairs     = zip yTrueList yPredList
        n         = if null pairs then 0 else 1 + maximum (map fst pairs ++ map snd pairs)
    in [ [ length [ () | (t,p) <- pairs, t == i, p == j ]
         | j <- [0..n-1] ]
       | i <- [0..n-1] ]


printConfusionMatrix
  :: [String]   -- labels
  -> [[Int]]    -- confusion matrix
  -> IO ()
printConfusionMatrix labels m = do
  let n        = length m
      lbls     = if length labels == n then labels else map show [0..n-1]
      colW     = 10
      widthExp = n * colW                    
      padExp   = (widthExp - length "Expected") `div` 2
      padL     = colW + padExp

  putStrLn $ replicate padL ' ' ++ "Expected"

  putStr   $ replicate colW ' ' 
  forM_ lbls $ \l -> putStr $ printf "%*s" colW l
  putStrLn ""

  forM_ (zip lbls m) $ \(lab,row) -> do
    putStr $ printf "%*s" colW lab
    forM_ row $ \x -> putStr $ printf "%*d" colW x
    putStrLn ""
    


precision :: Tensor -> Tensor -> Float
precision yTrue yPred =
    let (tp, fp, _, _) = binaryConfusionMatrix yTrue yPred
    in fromIntegral tp / fromIntegral (tp + fp)

recall :: Tensor -> Tensor -> Float
recall yTrue yPred =
    let (tp, _, _, fn) = binaryConfusionMatrix yTrue yPred
    in fromIntegral tp / fromIntegral (tp + fn)

f1Score :: Tensor -> Tensor -> Float
f1Score yTrue yPred =
    let p = precision yTrue yPred
        r = recall yTrue yPred
    in 2 * (p * r) / (p + r)

-- microF1Score :: 

-- macroF1Score :: 

-- weightedF1Score :: 



