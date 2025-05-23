module Preprocess where 


import Codec.Binary.UTF8.String (encode) 
import GHC.Generics
import qualified Data.ByteString.Lazy as B 

import Data.Word (Word8)
import qualified Data.Map.Strict as M 
import Data.List (nub)
import qualified Data.ByteString.Lazy.Char8 as BL
import qualified Data.Aeson as Aeson
import Torch hiding (min,max)
import Data.Char (toLower)
import qualified Torch.Functional.Internal as FI

allowedChars :: [Char]
allowedChars = "abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüæœ "

isAllowedChar :: Word8 -> Bool
isAllowedChar w =
  let c = head (encode [toLower (toEnum (fromEnum w))])
  in c `elem` (map (head . encode . (:[])) allowedChars)



-- 'abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüæœ'


preprocess ::
  B.ByteString -> -- input
  [B.ByteString]  -- wordlist per line
preprocess texts =
  let cleaned  = B.pack $ filter isAllowedChar (B.unpack texts)
      lowered  = BL.map toLower cleaned
  in  B.split (head $ encode " ") lowered

wordToIndexFactory ::
  [B.ByteString] ->     -- wordlist
  (B.ByteString -> Int) -- function converting bytestring to index (unknown word: 0)
wordToIndexFactory wordlst wrd = M.findWithDefault (length wordlst) wrd (M.fromList (zip wordlst [0..]))


saveToJson :: [B.ByteString] -> String -> IO ()
saveToJson wList path = do
  let wordlstStr = map BL.unpack wList
  BL.writeFile path (Aeson.encode wordlstStr)


loadFromJson :: String -> IO [B.ByteString]
loadFromJson path = do
  jsonData <- BL.readFile path
  case Aeson.decode jsonData of
    Just wordList -> return $ map BL.pack wordList
    Nothing -> do
      putStrLn $ "Error: Could not decode JSON from " ++ path
      return []



createPair 
  :: [Int]         -- ^ suite d’indices de mots
  -> Int           -- ^ taille de la fenêtre (window size)
  -> [(Int, Int)]  -- ^ liste de (mot_centre, mot_contexte)
createPair wordsIdx windowSize = concatMap pairsAt (zip [0..] wordsIdx)
  where
    n = length wordsIdx
    pairsAt (i, w) =
      let start = max 0         (i - windowSize)
          end   = min (n - 1)   (i + windowSize)
      in [ (w, wordsIdx !! j)
         | j <- [start .. end], j /= i ]


createDataloader 
  :: [(Int, Int)]   -- ^ liste de (mot_centre, mot_contexte)
  -> Int            -- ^ taille du mini-batch
  -> [(Tensor,Tensor)] -- (x,y) [((B,1),(B,1))]
createDataloader pairs batchSize =
  let batches   = chunksOf batchSize pairs
      makeBatch grp =
        let (xs, ys) = unzip grp
            x =  (asTensor xs) 
            y =  (asTensor ys) 
        in (x, y)
  in map makeBatch batches


chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs =
  let (f, r) = splitAt n xs
  in f : chunksOf n r