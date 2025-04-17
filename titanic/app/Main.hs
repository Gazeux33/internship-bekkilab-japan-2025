module Main (main) where

import LoadData (loadCSV,dataToTensors,createDataloader)
import Model (MLPSpec(..),processTrain)
import Torch
import Lib

main :: IO ()
main = do 
    putStrLn "Start..."
    csvResult <- loadCSV "data/train_preprocessed.csv"
    case csvResult of
        Left err -> putStrLn err
        Right rows -> do
            let tensor@(x_tensor,y_tensor) = dataToTensors rows
            putStrLn $ "Vector Size : " ++ show ( shape x_tensor ) ++ "," ++ show ( shape y_tensor )
            let dataloader = createDataloader tensor 16
            putStrLn $ "Dataloader Size : " ++ show (length dataloader )
            putStrLn $ "Size of one item : (" ++ show (shape ( fst (head dataloader ))) ++ "),(" ++ show (shape ( snd (head dataloader ))) ++ ")"
            
            
            initialModel <- sample (MLPSpec 7 16 8 1) 
            let optimizer = GD
            let lr = 0.1
            
            
            putStrLn $ "Start Train"
            (model, loss) <- processTrain dataloader initialModel optimizer lr 10
            putStrLn $ "End Train"

