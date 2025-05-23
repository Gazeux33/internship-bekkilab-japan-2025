{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Embedding where 

import Torch
import Torch.Functional as F
import GHC.Generics

data EmbeddingSpec = EmbeddingSpec {
  wordNum :: Int, -- the number of words
  wordDim :: Int  -- the dimention of word embeddings
} deriving (Show, Eq, Generic)

data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)

instance Randomizable EmbeddingSpec Embedding where
  sample EmbeddingSpec {..} = do
    wTensor <- randIO' [wordNum, wordDim]
    wordEmbedding <- makeIndependent wTensor
    return $ Embedding wordEmbedding

embeddingForward
  :: Embedding
  -> Tensor
  -> Tensor
embeddingForward Embedding{..} input = F.embedding' (toDependent wordEmbedding) input

     