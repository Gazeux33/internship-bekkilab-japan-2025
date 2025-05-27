{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

module Model where


import GHC.Generics
import Torch
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.NN.Recurrent.Cell.LSTM 


data ModelSpec = ModelSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


data Model = Model {
  emb :: Embedding,
  lstm :: LSTMCell
} deriving (Show, Generic, Parameterized)

instance Randomizable ModelSpec Model where
  sample ModelSpec {..} =
    Model
      <$> (Embedding <$> (makeIndependent =<< randnIO' [wordDim, wordNum]))
      <*> sample (LSTMSpec 10 10)

initialize ::
  ModelSpec ->
--   FilePath ->
  IO Model
initialize modelSpec = do
  randomizedModel <- sample modelSpec
  --loadedEmb <- loadParams (emb randomizedModel) embPath
  return randomizedModel