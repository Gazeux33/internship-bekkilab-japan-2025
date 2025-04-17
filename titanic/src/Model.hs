{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Model
    ( MLPSpec(..),processTrain

    ) where

import Control.Monad (foldM,when)
import GHC.Generics (Generic)
import Torch
import qualified Torch.Functional.Internal as FI
import qualified Torch.Functional as F


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
  let loss = F.binaryCrossEntropyLoss' label output 
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)


processTrainEpoch :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> Float -> Int -> IO (MLP, Float)
processTrainEpoch dataloader model optimizer lr epoch = do
    putStrLn $ "Starting Epoch " ++ show epoch
    foldM (\(m', _) (i, batch) -> do
                        (newModel, loss) <- processBatch m' optimizer lr batch
                        when (i `mod` 10 == 0) $
                          putStrLn $ "Epoch " ++ show epoch ++ ", Batch " ++ show i ++ " : loss : " ++ show loss
                        pure (newModel, loss)
                ) (model, 0.0) (zip [1..] dataloader)

processTrain :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> Float -> Int -> IO (MLP, Float)
processTrain dataloader model optimizer lr epochs = do
    foldM (\(m, _) epoch -> processTrainEpoch dataloader m optimizer lr epoch) (model, 0.0) [1..epochs]