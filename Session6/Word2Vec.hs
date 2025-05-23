{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE StandaloneDeriving #-}

module Main (main) where

import qualified Data.ByteString.Lazy as B 

import Torch hiding (take)

import Model
import Embedding
import MLP
import Preprocess
import Config 
import Evaluation

main :: IO ()
main = do

  -- load word list
  wordlst <- loadFromJson vocabPath
  let wordToIndex = wordToIndexFactory wordlst
      vocabSize = length wordlst
  putStrLn  $ "Vocab Size : " ++ show vocabSize

  -- create model
  embdLayer <- sample $ (EmbeddingSpec (vocabSize+1) embdDim)
  mlpModel <- sample $ (MLPSpec embdDim (1024) (2048) (2048) (vocabSize+1))
  let initialModel = Model embdLayer mlpModel

   -- create optimizer
  let optimizer = GD

  -- Load Eval File
  evalData <- loadEvalFile evalFilePath

  -- Load the text
  texts <- B.readFile midTextFilePath
  let idxes = map wordToIndex (preprocess texts)
      pairs = createPair idxes windowSize -- [(Center,Context)] [(Int,Int)]
      dataloader = createDataloader pairs batchSize -- [(Tensor,Tensor)]
  putStrLn $ "Dataloader Size: " ++ show (length dataloader)

  putStrLn "Start Train"
  _ <- trainWithLossTracking dataloader initialModel evalData wordToIndex optimizer learningRate epoch printFreq saveFreq evalFreq embdModelPath
  putStrLn "End Train"







  




  