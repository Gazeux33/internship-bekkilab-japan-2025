{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Model
    ( MLPSpec(..),processTrain

    ) where

{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}


import LoadData (Batch)
import Control.Monad (foldM)
import GHC.Generics (Generic)
import Torch
import qualified Torch.Functional.Internal as FI


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

    linear l2 . relu . linear l1 . relu . linear l0


processBatch :: Optimizer o => MLP -> o -> Float -> (Tensor, Tensor) -> IO (MLP, Float)
processBatch model optimizer lr (input, label) = do
  let output = mlp model input
  let loss = FI.mse_loss output label 1  
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)


evalBatch :: MLP  -> (Tensor, Tensor) -> Float
evalBatch model optimizer lr (input, label) = do
  let output = mlp model input
  let loss = FI.mse_loss output label 1  
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)
  


processTrain :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> Float -> IO (MLP, Float)
processTrain dataloader model optimizer lr = do
  foldM (\(m, _) (i, batch) -> do
            (newModel, loss) <- processBatch m optimizer lr batch
            putStrLn $ "Batch " ++ show i ++ " : loss : " ++ show loss
            pure (newModel, loss)
        ) (model, 0.0) (zip [1..] dataloader)


processEval  ::  [(Tensor, Tensor)] -> MLP
processEval dataloader = do




