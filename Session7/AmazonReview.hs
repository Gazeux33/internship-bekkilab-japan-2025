{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}


module Main (main) where
import qualified Data.ByteString.Lazy as B
import Codec.Binary.UTF8.String (encode)


import Model
import Preprocess 

amazonReviewPath :: FilePath
amazonReviewPath = "data/Amazon/train.jsonl"

wordLstPath = "data/embedding/vocab.txt"

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


  initModel <- initialize (ModelSpec 9 9)

  putStrLn $ "end of main"





-- createDataloaders :: [AmazonReview] -> [(Tensor,Tensor)]