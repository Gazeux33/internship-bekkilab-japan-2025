module Main (main) where

import Torch
import Data (loadData)
import Model (trainModel,evalModel)
import ML.Exp.Chart 






main :: IO ()
main = do
    putStrLn "Start..."
    
    result1 <- loadData "data/train.csv"
    result2 <- loadData "data/eval.csv"
    result3 <- loadData "data/valid.csv"
    
    case (result1, result2, result3) of
        (Right (x_train, y_train), Right (x_eval, y_eval), Right (x_valid, y_valid)) -> do
            putStrLn $ "x_train:" ++ show (length x_train) ++ " y_train:" ++ show (length y_train)
            putStrLn $ "x_eval:" ++ show (length x_eval) ++ " y_eval:" ++ show (length y_eval)
            putStrLn $ "x_valid:" ++ show (length x_valid) ++ " y_valid:" ++ show (length y_train)

            putStrLn "Start train"

            (final_a, final_b, losses) <- trainModel x_train y_train x_valid y_valid
            putStrLn $ "Final A: " ++ show final_a
            putStrLn $ "Final B: " ++ show final_b
            putStrLn $ "Final loss: " ++ show (evalModel x_eval y_eval (final_a, final_b))

            putStrLn "End train "

            -- Plot the results with actual loss values
            putStrLn "Plot Result"
            let outputPath = "outpout/train.png"
            drawLearningCurve outputPath "Linear Regression Training" [("Validation Loss", losses)]
            putStrLn $ "Learning curve saved to " ++ outputPath

        (Left err, _, _) -> putStrLn $ "Error loading training data: " ++ err
        (_, Left err, _) -> putStrLn $ "Error loading evaluation data: " ++ err
        (_, _, Left err) -> putStrLn $ "Error loading validation data: " ++ err
