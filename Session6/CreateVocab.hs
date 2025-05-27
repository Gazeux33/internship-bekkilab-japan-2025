module Main (main) where

import Data.List (sortBy)
import Data.Ord  (comparing)
import qualified Data.Map.Strict as M
import qualified Data.ByteString.Lazy as B 
import qualified Data.ByteString.Lazy.Char8 as C8

import Preprocess
import Config (fullTextFilePath, vocabPath,maxVocab,smallTextFilePath)

main :: IO ()
main = do
  -- Load the text
  texts <- B.readFile fullTextFilePath


  let wordLines = preprocess texts
      freqMap = M.fromListWith (+) [ (w, 1) | w <- wordLines ]
      sortedWords = map fst
                  $ take maxVocab
                  $ sortBy (flip (comparing snd))
                          (M.toList freqMap)
      wordlst = map C8.unpack sortedWords 

  writeFile vocabPath (unlines wordlst)
  putStrLn $ "Save to " ++ vocabPath
