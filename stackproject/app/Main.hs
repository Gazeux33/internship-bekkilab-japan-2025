

module Main where

import LoadData (loadCSV, dataToTensors,createDataset,createDataloader,Dataloader,processEval)
import Model (MLPSpec(..), processTrain)
import Torch
import Data.Vector()
import Data.Either() 


batchSize :: Int 
batchSize = 16

winSize :: Int
winSize = 7

loadData :: String -> IO (Either String Dataloader)
loadData filePath = do
    csvResult <- loadCSV filePath
    case csvResult of
        Left err -> return (Left err)
        Right rows -> do
            let vector = dataToTensors rows
            let dataset = createDataset vector winSize
            let dataloader = createDataloader dataset batchSize
            return (Right dataloader)



main :: IO ()
main = do
    putStrLn "Start..."
    trainData <- loadData "data/train.csv"
    case trainData of
        Left err -> putStrLn $ "Erreur de chargement CSV: " ++ err
        Right dataloader -> do
            putStrLn $ " train dataloader size:  " ++ show (length dataloader)
            model <- sample (MLPSpec 7 8 4 1)
            let optimizer = GD
            let lr = 0.00001

            putStrLn $ "Start Train"
            (model,loss) <- processTrain dataloader model optimizer lr
            putStrLn $ "End Train"

    evalData <- loadData "data/eval.csv"
    case evalData of
        Left err -> putStrLn $ "Erreur de chargement CSV: " ++ err
        Right dataloader -> do
            putStrLn $ "eval dataloader size:  " ++ show (length dataloader)

            putStrLn $ "Start eval"
            (model,loss) <- processEval dataloader model optimizer lr
            putStrLn $ "End eval"

    
                

    putStrLn $ "End....  "


