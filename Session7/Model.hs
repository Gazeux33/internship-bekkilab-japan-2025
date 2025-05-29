{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

module Model where

import MLP
import Torch.Layer.NonLinear (ActName(..))
import Torch.Device (Device(..))
import GHC.Generics
import Torch
import Torch.NN (Parameter, Parameterized(..), Randomizable(..))
import Torch.Layer.RNN

data ModelSpec = ModelSpec {
  wordNum :: Int,      -- nombre de mots dans le vocabulaire
  wordDim :: Int,      -- dimension des embeddings
  hiddenDim :: Int,    -- dimension cachée du RNN
  numLayers :: Int,    -- nombre de couches RNN
  bidirectional :: Bool, -- RNN bidirectionnel ou non
  device :: Device     -- device (CPU/CUDA)
} deriving (Show, Eq, Generic)


data Embedding = Embedding {
    wordEmbedding :: Parameter
  } deriving (Show, Generic, Parameterized)


data Model = Model {
  emb :: Embedding,
  rnn :: RnnParams,
  initialStates :: InitialStatesParams,
  mlp :: MLP
} deriving (Show, Generic, Parameterized)

instance Randomizable ModelSpec Model where
  sample ModelSpec {..} = do
    -- Créer les embeddings
    embedding <- Embedding <$> (makeIndependent =<< randnIO' [wordNum, wordDim])
    
    -- Créer les paramètres du RNN
    let rnnHypParams = RnnHypParams {
          dev = device,
          bidirectional = bidirectional,
          inputSize = wordDim,    -- dimension des embeddings en entrée
          hiddenSize = hiddenDim, -- dimension cachée
          numLayers = numLayers,  -- nombre de couches
          hasBias = True          -- utiliser les biais
        }
    rnnParams <- sample rnnHypParams
    
    -- Créer les états initiaux
    let initialStatesHypParams = InitialStatesHypParams {
          dev = device,
          bidirectional = bidirectional,
          hiddenSize = hiddenDim,
          numLayers = numLayers
        }
    initialStatesParams <- sample initialStatesHypParams

    mlp <- sample (MLPSpec (if bidirectional then hiddenDim*2 else hiddenDim) 1) -- MLP avec une couche cachée et une sortie
    
    return $ Model embedding rnnParams initialStatesParams mlp


forwardModel :: Model -> Tensor -> (Tensor, Tensor,Tensor)
forwardModel Model{..} inputIds =
  let -- Récupérer les embeddings en appliquant le champ wordEmbedding sur emb
      embeddings = embedding' (toDependent $ wordEmbedding emb) inputIds
      
      -- Passer par le RNN avec activation tanh
      (output, hiddenStates) = rnnLayers rnn Tanh Nothing (toDependent $ h0s initialStates) embeddings

      logits = mlpForward mlp output
      
  in (output, hiddenStates , logits)


initialize ::
  ModelSpec ->
--   FilePath ->
  IO Model
initialize modelSpec = do
  randomizedModel <- sample modelSpec
  --loadedEmb <- loadParams (emb randomizedModel) embPath
  return randomizedModel