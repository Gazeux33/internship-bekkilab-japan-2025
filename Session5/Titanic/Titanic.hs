module Main (main) where

import LoadData (loadCSV,dataToTensors,createDataloader)
import Model 
import Torch
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools


epoch :: Int
epoch = 3000

batchSize :: Int
batchSize = 32

learningRate :: Float
learningRate = 0.0001

main :: IO ()
main = do 
    putStrLn "Start..."
    csvResult <- loadCSV "data/titanic/train_preprocessed.csv"
    case csvResult of
        Left err -> putStrLn err
        Right rows -> do
            let tensor@(x_tensor,y_tensor) = dataToTensors rows
            putStrLn $ "Vector Size : " ++ show ( shape x_tensor ) ++ "," ++ show ( shape y_tensor )
            let dataloader = createDataloader tensor batchSize
            putStrLn $ "Dataloader Size : " ++ show (length dataloader )
            putStrLn $ "Size of one item : (" ++ show (shape ( fst (head dataloader ))) ++ "),(" ++ show (shape ( snd (head dataloader ))) ++ ")"
            
            
            initialModel <- sample (MLPSpec 7 8 8 1) 
            let optimizer = mkAdam 10 0.9 0.999 (flattenParameters initialModel)

            
            
            putStrLn $ "Start Train"
            (finalModel, trainLosses) <- trainWithLossTracking dataloader initialModel optimizer learningRate epoch 20

            drawLearningCurve "Session5/Titanic/output/titanic-train-curve.png" "Titanic Learning Curve" 
                              [("Training Loss", reverse trainLosses)
                               ]
            putStrLn $ "End Train"

