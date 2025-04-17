module Main where

import LoadData (loadCSV, dataToTensors,createDataset,createDataloader,Dataloader)
import Model (MLPSpec(..), processTrain,processEval)
import Torch
import Data.Vector()
import Data.Either() 
import Torch.Serialize (saveParams,loadParams)



batchSize :: Int 
batchSize = 16

winSize :: Int
winSize = 7

loadModel :: Bool
loadModel = True

savePath :: FilePath
savePath = "output/trainedModel"

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
    
    initialModel <- sample (MLPSpec 7 16 8 1) 
    let optimizer = GD
    let lr = 0.00001

    initialModel <- if loadModel then loadParams initialModel savePath else return initialModel
    
    trainedModel <- case trainData of
        Left err -> do
            putStrLn $ "Erreur de chargement CSV: " ++ err
            return initialModel  
        Right dataloader -> do
            putStrLn $ " train dataloader size:  " ++ show (length dataloader)
            
            putStrLn $ "Start Train"
            (model, loss) <- processTrain dataloader initialModel optimizer lr 2
            putStrLn $ "End Train"
            return model  
    
    evalData <- loadData "data/eval.csv"
    case evalData of
        Left err -> putStrLn $ "Erreur de chargement CSV: " ++ err
        Right dataloader -> do
            putStrLn $ "eval dataloader size:  " ++ show (length dataloader)

            putStrLn $ "Start eval"
            processEval dataloader trainedModel  
            putStrLn $ "End eval"


    saveParams trainedModel savePath

    putStrLn $ "End....  "