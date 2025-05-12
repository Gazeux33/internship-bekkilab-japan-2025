module Main (main) where

import Torch
import Data (loadData)
import Model 
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools


batchSize :: Int
batchSize = 16

learningRate :: Float
learningRate = 0.0001

epoch :: Int
epoch = 30

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
            
            initialModel <- sample (MLPSpec 7 16 16 1) 
            let optimizer = mkAdam 0 0.9 0.999 (flattenParameters initialModel)
            
            putStrLn $ "***Start Train***"
            (finalModel, trainLosses, validLosses) <- trainWithLossTracking trainDataloader validDataloader initialModel optimizer learningRate epoch 10


            drawLearningCurve "Session5/AdmissionChances/output/admission-train-curve.png" "Admission Chances Learning Curve" 
                              [("Training Loss", reverse trainLosses),
                              ("Validation Loss", reverse validLosses)
                               ]

            
            putStrLn $ "Learning curve saved to admission-learning-curve.png"

            putStrLn $ "***Start Test***"
            evalLoss <- processTest finalModel evalDataloader
            putStrLn $ "Final Eval Loss: " ++ show evalLoss

        (Left err, _, _) -> putStrLn $ "Error loading training data: " ++ err
        (_, Left err, _) -> putStrLn $ "Error loading evaluation data: " ++ err
        (_, _, Left err) -> putStrLn $ "Error loading validation data: " ++ err
