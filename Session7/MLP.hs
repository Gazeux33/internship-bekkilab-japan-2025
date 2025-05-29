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
    outputFeatures :: Int
  }
  deriving (Show, Eq)


data MLP = MLP
  { l0 :: Linear
  }
  deriving (Generic, Show, Parameterized)


instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} =
    MLP
      <$> sample (LinearSpec inputFeatures outputFeatures)


mlpForward :: MLP -> Tensor -> Tensor
mlpForward MLP{..} input =
  let logits = linear l0 input
  in  logits




