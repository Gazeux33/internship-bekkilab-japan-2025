module Main where

import LoadData (loadCSV, dataToTensors,createDataset,createDataloader)
import Torch
import Data.Vector()
import Data.Either()


batchSize :: Int 
batchSize = 16

winSize :: Int
winSize = 7

loadData :: String -> IO (Either String [[(Tensor, Tensor)]])
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
    dataResult <- loadData "data/eval.csv"
    case dataResult of
        Left err -> putStrLn $ "Erreur de chargement CSV: " ++ err
        Right dataloader -> do
            putStrLn $ "dataloader size:  " ++ show (length dataloader)


