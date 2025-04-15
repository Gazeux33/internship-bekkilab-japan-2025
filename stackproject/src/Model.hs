{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module Model
    ( MLPSpec(..),processTrain,processEval

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

    linear l2  . linear l1  . linear l0


processBatch :: Optimizer o => MLP -> o -> Float -> (Tensor, Tensor) -> IO (MLP, Float)
processBatch model optimizer lr (input, label) = do
  let output = mlp model input
  let loss = FI.mse_loss output label 1  
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)


evalBatch :: MLP  -> (Tensor, Tensor) -> Float
evalBatch model  (input, label) = asValue $ FI.mse_loss output label 1  
    where output = mlp model input

  


processTrainEpoch :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> Float -> Int -> IO (MLP, Float)
processTrainEpoch dataloader model optimizer lr epoch = do
    putStrLn $ "Starting Epoch " ++ show epoch
    foldM (\(m', _) (i, batch) -> do
                        (newModel, loss) <- processBatch m' optimizer lr batch
                        putStrLn $ "Epoch " ++ show epoch ++ ", Batch " ++ show i ++ " : loss : " ++ show loss
                        pure (newModel, loss)
                ) (model, 0.0) (zip [1..] dataloader)

processTrain :: Optimizer o => [(Tensor, Tensor)] -> MLP -> o -> Float -> Int -> IO (MLP, Float)
processTrain dataloader model optimizer lr epochs = do
    foldM (\(m, _) epoch -> processTrainEpoch dataloader m optimizer lr epoch) (model, 0.0) [1..epochs]

processEval :: [(Tensor, Tensor)] -> MLP -> IO ()
processEval dataloader model = do
    mapM_ (\(i, batch) -> do
                        let loss = evalBatch model batch
                        putStrLn $ "Batch " ++ show i ++ " : loss : " ++ show loss
                ) (zip [1..] dataloader)




