module Main (main) where

import Torch (sample,loadParams)

import Preprocess
import Embedding
import Config (embdModelPath, vocabPath, evalFilePath, predictionPath, embdDim)
import Evaluation

main :: IO ()
main = do

    -- Load vocab
    wordlst <- loadFromJson vocabPath
    let wordToIndex = wordToIndexFactory wordlst
        vocabSize = length wordlst
    putStrLn  $ "Vocab Size : " ++ show vocabSize

    -- Load Model
    embdLayer <- sample $ (EmbeddingSpec (vocabSize+1) embdDim)
    loadedEmdb <- loadParams embdLayer embdModelPath
    putStrLn $ "Embd loaded from " ++ embdModelPath

    -- Make and save predictions
    savePredictions loadedEmdb wordlst predictionPath

    -- Load and filter the evaluation file
    filteredContents <- loadEvalFile evalFilePath

    -- Evaluate the model
    score <- evalModel filteredContents loadedEmdb wordToIndex
    putStrLn $ "Score: " ++ show score


