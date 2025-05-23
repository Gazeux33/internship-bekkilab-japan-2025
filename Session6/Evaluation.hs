module Evaluation where

import Torch hiding (take,abs,cosineSimilarity)

import Data.List.Split (splitOn)
import qualified Data.ByteString.Lazy as B 
import qualified Data.ByteString.Lazy.Char8 as BC
import qualified Torch.Functional.Internal as FI

import Preprocess
import Embedding
import Config

savePredictions:: Embedding -> [a] -> String -> IO ()
savePredictions model vocab path = do
    let input = toDType Int64 (arange' 0 (length vocab) 1)
        predictions = embeddingForward model input
    save [predictions] path
    putStrLn $ "Predictions if shape " ++ show (shape predictions) ++ " saved to " ++ path

readTSV :: FilePath -> IO [[String]]
readTSV filePath = do
    contents <- readFile filePath
    let rows = lines contents
        parsedRows = map (splitOn "\t") rows
    return parsedRows

loadEvalFile :: String -> IO  [[String]] 
loadEvalFile path = do
    contents <- readTSV path
    let filteredContents = filter (\xs -> case xs of
                                          [] -> False
                                          (x:_) -> x /= "") contents
    return filteredContents

evalOneLine :: [String] -> Embedding -> (B.ByteString -> Int) -> (Int, Float, Int)
evalOneLine line embdLayer wordToIndex = (diff,cs,ds)
    where 
          number = read (line !! 0) :: Int
          s1 = line !! 1
          s2 = line !! 2
          s1Bytes = BC.pack s1
          s2Bytes = BC.pack s2
          s1Indices = map wordToIndex (preprocess s1Bytes)
          s2Indices = map wordToIndex (preprocess s2Bytes)
          s1Tensor = toDType Int64 (asTensor s1Indices)
          s2Tensor = toDType Int64 (asTensor s2Indices)
          pred1 = embeddingForward embdLayer s1Tensor
          pred2 = embeddingForward embdLayer s2Tensor
          mean1 = FI.meanDim pred1 0 False Float
          mean2 = FI.meanDim pred2 0 False Float
          cs = cosineSimilarity mean1 mean2
          ds = discretizeScore cs
          diff = getDifference number ds

getDifference :: Int -> Int -> Int
getDifference x y = abs (x - y)
        
cosineSimilarity :: Tensor -> Tensor -> Float
cosineSimilarity t1 t2 = asValue $ dotProduct / (normX1 * normX2)
    where dotProduct = dot t1 t2
          normX1 = FI.sqrt $ dot t1 t1
          normX2 = FI.sqrt $ dot t2 t2

discretizeScore :: Float -> Int
discretizeScore x 
    | x >= -1.0 && x <= -0.6 = 0
    | x >= -0.5 && x <= 0.0  = 1
    | x >= 0.1  && x <= 0.3  = 2
    | x >= 0.4  && x <= 0.6  = 3
    | x >= 0.7  && x <= 0.8  = 4
    | x >= 0.9  && x <= 1.0  = 5
    | otherwise              = error "Value out of range [-1.0, 1.0]"

evalModel :: [[String]] -> Embedding -> (B.ByteString -> Int) -> IO Int
evalModel evalData embdLayer wordToIndex = do
    results <- mapM (\l -> do
        let (diff, cs, ds) = evalOneLine l embdLayer wordToIndex
        putStrLn $ "Line: " ++ (l !! 0) ++ ", Difference: " ++ show diff ++ 
                  ", Cosine Similarity: " ++ show cs ++ ", Discretized Score: " ++ show ds
        return diff) evalData
    let totalDiff = sum results
    putStrLn $ "Total difference: " ++ show totalDiff
    return totalDiff


    


