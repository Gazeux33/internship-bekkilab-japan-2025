module Main (main) where

import Torch
import Data (loadData)
import Model (MLPSpec(..),processTrain,processTest)


batchSize :: Int
batchSize = 32

learningRate :: Float
learningRate = 0.001

epoch :: Int
epoch = 10000

main :: IO ()
main = do
    
    result1 <- loadData "data/train.csv" batchSize
    result2 <- loadData "data/eval.csv" batchSize
    result3 <- loadData "data/valid.csv" batchSize
    
    case (result1, result2, result3) of
        (Right trainDataloader, Right evalDataloader, Right validDataloader) -> do
            putStrLn $ "train_dataloader:" ++ show (length trainDataloader) 
            putStrLn $ "eval_dataloader:" ++ show (length evalDataloader) 
            putStrLn $ "valid_dataloader:" ++ show (length validDataloader) 



            let optimizer = GD
            initialModel <- sample (MLPSpec 7 32 16 1) 
            
            putStrLn $ "***Start Train***"
            (model, loss) <- processTrain trainDataloader validDataloader initialModel optimizer learningRate epoch 1000


            putStrLn $ "***Start Test***"
            evalLoss <- processTest model evalDataloader
            putStrLn $ "Final Eval Loss: " ++ show evalLoss

        (Left err, _, _) -> putStrLn $ "Error loading training data: " ++ err
        (_, Left err, _) -> putStrLn $ "Error loading evaluation data: " ++ err
        (_, _, Left err) -> putStrLn $ "Error loading validation data: " ++ err
