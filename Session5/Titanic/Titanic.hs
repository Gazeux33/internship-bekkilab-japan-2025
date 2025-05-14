module Main (main) where

import LoadData (loadCSV,dataToTensors,createDataloader)
import Model 
import Torch
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools
import Evaluation


epoch :: Int
epoch = 2000

batchSize :: Int
batchSize = 64

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
            let dataloader = createDataloader tensor batchSize
            putStrLn $ "Dataloader Size : " ++ show (length dataloader )
            
            
            initialModel <- sample (MLPSpec 7 16 16 1) 
            let optimizer = mkAdam 10 0.9 0.999 (flattenParameters initialModel)

            
            
            putStrLn $ "Start Train..."
            (finalModel, trainLosses) <- trainWithLossTracking dataloader initialModel optimizer learningRate epoch 1000

            drawLearningCurve "Session5/Titanic/output/titanic-train-curve.png" "Titanic Learning Curve" 
                              [("Training Loss", reverse trainLosses)
                               ]


            let (target,pred) = finalPrediction finalModel dataloader
            let roundPreds = roundTensor pred
            let roundTargets = roundTensor target

            let acc = accuracy roundTargets roundPreds
                confusion = multiclassConfusionMatrix roundTargets roundPreds
                pre = precision roundTargets roundPreds
                recallModel = recall roundTargets roundPreds
                f1 = f1Score roundTargets roundPreds

            

            putStrLn $ "Final Accuracy: " ++ show acc
            putStrLn $ "" 
            printConfusionMatrix ["Die" , "Survive"] confusion
            putStrLn $ "" 
            putStrLn $ "Final Precision: " ++ show pre
            putStrLn $ "Final Recall: " ++ show recallModel
            putStrLn $ "Final F1 Score: " ++ show f1
            
            putStrLn $ "End Train"

