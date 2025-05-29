{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


module Main (main) where
import qualified Data.ByteString.Lazy as B
import Codec.Binary.UTF8.String (encode)

import Torch
import Model
import Preprocess 
import Training

amazonReviewPath :: FilePath
amazonReviewPath = "data/Amazon/valid.jsonl"

wordLstPath = "data/embedding/vocab.txt"

batchSize :: Int
batchSize = 32

embeddingDim :: Int
embeddingDim = 128

hDim :: Int
hDim = 128

nLayers :: Int
nLayers = 2

main :: IO ()
main = do
  jsonl <- B.readFile amazonReviewPath
  let amazonReviews = decodeToAmazonReview jsonl
  let reviews = case amazonReviews of
                  Left err -> []
                  Right reviews -> reviews
  putStrLn $ "Number of reviews: " ++ show (length reviews)
  
  -- load word list (It's important to use the same list as whan creating embeddings)
  wordLst <- fmap (B.split (head $ encode "\n")) (B.readFile wordLstPath)
  let wordToIndex = wordToIndexFactory wordLst
      dataset = createDataset reviews wordToIndex
      vocabSize = length wordLst
  putStrLn $ "Number of dataset: " ++ show (length dataset)

  let (x,y) = head dataset
  putStrLn $ "x shape: " ++ show (shape x)
  putStrLn $ "y shape: " ++ show (shape y)



  initModel <- initialize (ModelSpec (vocabSize+1) embeddingDim hDim nLayers True (Device CPU 0))

  let (output,hiddenState,logits) = forwardModel initModel x


  (newModel, loss) <- trainWithLossTracking dataset initModel GD 0.01 2 1
  putStrLn $ "Loss after first batch: " ++ show loss



  putStrLn $ "end of main"





-- createDataloaders :: [AmazonReview] -> [(Tensor,Tensor)]