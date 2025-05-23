{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}

module MLP where

import qualified Torch.Functional.Internal as FI
import Control.Monad (foldM,when)
import qualified Torch.Functional as F
import GHC.Generics (Generic)
import Torch

data MLPSpec = MLPSpec
  { inputFeatures :: Int,
    hiddenFeatures0 :: Int,
    hiddenFeatures1 :: Int,
    hiddenFeatures2 :: Int,
    outputFeatures :: Int
  }
  deriving (Show, Eq)


data MLP = MLP
  { l0 :: Linear,
    l1 :: Linear,
     l2 :: Linear,
    l3 :: Linear
  }
  deriving (Generic, Show, Parameterized)


instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec inputFeatures hiddenFeatures0)
      <*> sample (LinearSpec hiddenFeatures0 hiddenFeatures1)
      <*> sample (LinearSpec hiddenFeatures1 hiddenFeatures2)
      <*> sample (LinearSpec hiddenFeatures2 outputFeatures)

mlpForward :: MLP -> Tensor -> Tensor
mlpForward MLP{..} input =
  let h0     = F.relu $ linear l0 input
      h1     = F.relu $ linear l1 h0
      h2     = F.relu $ linear l2 h1
      logits = linear l3 h2
  in  logits




