module Config where

fullTextFilePath :: String
fullTextFilePath = "data/embedding/full_text.txt"

midTextFilePath :: String
midTextFilePath = "data/embedding/mid_text.txt"

smallTextFilePath :: String
smallTextFilePath = "data/embedding/small_text.txt"

vocabPath :: String
vocabPath = "data/embedding/vocab.json"

embdModelPath :: String
embdModelPath =  "data/embedding/embedding_model.params"

fullModelPath :: String
fullModelPath = "data/embedding/full_model.params"

predictionPath :: String
predictionPath = "data/embedding/predictions_embed.pt"

evalFilePath :: String
evalFilePath = "data/embedding/answer-answer.test.tsv"

maxVocab :: Int
maxVocab = 30000

embdDim :: Int
embdDim = 128

seqLen :: Int 
seqLen = 64

batchSize :: Int
batchSize = 128

learningRate :: Float
learningRate = 0.005

epoch :: Int
epoch = 1

windowSize :: Int
windowSize = 2

printFreq :: Int
printFreq = 1

saveFreq :: Int
saveFreq = 500

evalFreq :: Int
evalFreq = 500