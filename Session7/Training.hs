module Training where

import Torch
import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import qualified Data.ByteString.Lazy as B 
import Control.Monad (foldM,when)
import GHC.Generics
import Torch
import MLP
import Model
import ML.Exp.Chart   (drawLearningCurve)

processBatch :: Optimizer o => Model -> o -> Float -> (Tensor, Tensor) -> IO (Model, Float)
processBatch model optimizer lr (input, label) = do
  let (output,hidden,logits) = forwardModel model input
  let loss = computeMSE logits label
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)



computeMSE :: Tensor -> Tensor -> Tensor
computeMSE output target = F.mseLoss target (FI.select output 0 (-1))



processTrainEpoch :: Optimizer o => 
                    [(Tensor, Tensor)]  -- Training data
                    -> Model     
                    -> o
                    -> Float
                    -> Int
                    -> Int
                    -> [Float]          -- Ajout: listes de losses accumulées        -- Ajout: scores accumulés
                    -> IO (Model, [Float])
processTrainEpoch dataloader model optimizer lr epoch printFreq existingLosses = do
    foldM (\(m', losses) (i, batch) -> do
        (newModel, lossVal) <- processBatch m' optimizer lr batch
        let newLosses = lossVal : losses

        -- print and draw curve at intervals
        when (i `mod` printFreq == 0) $ do
            putStrLn $ "Epoch " ++ show epoch ++ " | Batch " ++ show i ++ " | Loss: " ++ show lossVal
            drawLearningCurve "Session7/output/word2vec-training.png"
                              "Word2Vec Training Losses Curve"
                              [("Training Loss", reverse newLosses)]
                       
        pure (newModel, newLosses)
      ) (model, existingLosses) (zip [1..] dataloader)


trainWithLossTracking :: Optimizer o => 
                       [(Tensor, Tensor)]
                    -> Model
                    -> o
                    -> Float
                    -> Int
                    -> Int
                    -> IO (Model, [Float])
trainWithLossTracking trainData initialModel optimizer lr epochs printFreq = do
    foldM (\(m, allLosses) ep -> do
        (newModel, epochLosses) <-
           processTrainEpoch trainData m optimizer lr
                             ep printFreq
                             allLosses
        pure (newModel, epochLosses)
      ) (initialModel, []) [1..epochs]




  

