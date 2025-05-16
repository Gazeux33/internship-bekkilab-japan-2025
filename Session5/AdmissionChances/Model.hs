{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Model where

import qualified Torch.Functional.Internal as FI
import Control.Monad (foldM,when)
import qualified Torch.Functional as F
import GHC.Generics (Generic)
import Torch

data MLPSpec = MLPSpec
  { inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    outputFeatures :: Int
  }
  deriving (Show, Eq)


data MLP = MLP
  { l0 :: Linear,
    l1 :: Linear,
    l2 :: Linear
  }
  deriving (Generic, Show, Parameterized)


instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec inputFeatures hiddenFeatures0)
      <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
      <*> sample (LinearSpec hiddenFeatures1 outputFeatures)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} =
    sigmoid . linear l2 . relu . linear l1 . relu . linear l0




processBatch :: Optimizer o => MLP -> o -> Float -> (Tensor, Tensor) -> IO (MLP, Float)
processBatch model optimizer lr (input, label) = do
  let output = mlp model input
  let loss = F.mseLoss label output 
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)



processTrainEpoch :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> Float -> Int  -> IO (MLP, Float)
processTrainEpoch dataloader model optimizer lr epoch  = do
    foldM (\(m', _) (i, batch) -> do
                        (newModel, loss) <- processBatch m' optimizer lr batch
                        pure (newModel, loss)
                ) (model, 0.0) (zip [1..] dataloader)


processTest :: MLP -> [(Tensor,Tensor)] -> IO Float
processTest model dataloader = do
    losses <- mapM (\(input, label) -> do
                      let output = mlp model input
                      let loss = F.mseLoss label output
                      return $ asValue loss) dataloader
    return $ sum losses / fromIntegral (length losses)


-- bidju hidjam

finalPrediction :: MLP -> [(Tensor, Tensor)] -> (Tensor, Tensor) -- list of target list of pred  for all batches
finalPrediction model batches = 
  let inputs       = map fst batches
      targets      = map snd batches
      targetsCat   = F.flattenAll $ FI.cat targets 0 
      predsCat     = F.flattenAll $ FI.cat (map (mlp model) inputs) 0
  in (targetsCat, predsCat)

trainWithLossTracking :: Optimizer o => 
                       [(Tensor, Tensor)]   -- Training data
                    -> [(Tensor, Tensor)]   -- Validation data
                    -> MLP                  -- Initial model
                    -> o                    -- Optimizer
                    -> Float                -- Learning rate
                    -> Int                  -- Number of epochs
                    -> Int                  -- Print frequency
                    -> IO (MLP, [Float], [Float]) --final model, training losses, validation losses
trainWithLossTracking trainData validData initialModel optimizer lr epochs printFreq = do
    foldM (\(model, trainLosses, validLosses) epoch -> do
              (newModel, trainLoss) <- processTrainEpoch trainData model optimizer lr epoch
              validLoss <- processTest newModel validData
              
              when (epoch `mod` printFreq == 0) $
                  putStrLn $ "Epoch " ++ show epoch ++ " | Train Loss: " ++ show trainLoss 
                           ++ " | Valid Loss: " ++ show validLoss
              
              pure (newModel, trainLoss : trainLosses, validLoss : validLosses)
          ) (initialModel, [], []) [1..epochs]


crossEntropyLoss :: Tensor -> Tensor -> Tensor
crossEntropyLoss target output = 
  let
      weight = ones' [last (shape output)]
      loss = FI.cross_entropy_loss output target weight 1 (-100) 0.0

    in
      loss
  
