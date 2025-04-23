{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Model
    ( MLPSpec(..),processBatch,processTrain,processTest

    ) where

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

processTrain :: Optimizer o => [(Tensor, Tensor)] -> [(Tensor, Tensor)] -> MLP -> o -> Float -> Int -> Int -> IO (MLP, Float)
processTrain trainDataloader validDataloader model optimizer lr epochs print_freq = do
    foldM (\(m, _) epoch -> do
              (newModel, loss) <- processTrainEpoch trainDataloader m optimizer lr epoch
              validLoss <- processTest newModel validDataloader
              when (epoch `mod` print_freq == 0) $
                  putStrLn $ "Epoch " ++ show epoch ++ " | Cost: " ++ show loss ++ " | Valid Loss: " ++ show validLoss
              pure (newModel, loss)
          ) (model, 0.0) [1..epochs]

