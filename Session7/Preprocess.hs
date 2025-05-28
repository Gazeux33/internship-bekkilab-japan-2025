{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

module Preprocess where
    
import Torch hiding (take)
import Codec.Binary.UTF8.String (encode) -- add utf8-string to dependencies in package.yaml
import Data.Aeson (FromJSON(..), ToJSON(..), eitherDecode)
import qualified Data.ByteString.Lazy as B
import qualified Data.Map.Strict as M 
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Word (Word8)
import Data.Char (toLower)
import qualified Data.ByteString.Internal as B (c2w)
import GHC.Generics

data AmazonReview = AmazonReview {
  rating :: Float,
  title :: String,
  text :: String,
  images :: [Image],
  asin :: String,
  parent_asin :: String,
  user_id :: String,
  timestamp :: Int,
  verified_purchase :: Bool,
  helpful_vote :: Int
  } deriving (Show, Generic)


data Image = Image {
  small_image_url :: String,
  medium_image_url :: String,
  large_image_url :: String
} deriving (Show, Generic)

instance FromJSON Image
instance ToJSON Image


instance FromJSON AmazonReview
instance ToJSON AmazonReview


decodeToAmazonReview ::
  B.ByteString ->
  Either String [AmazonReview] 
decodeToAmazonReview jsonl =
  let jsonList = B.split (B.c2w '\n') jsonl
  in sequenceA $ map eitherDecode jsonList



createDataset :: [AmazonReview] -> (B.ByteString -> Int) -> [(Tensor, Tensor)]
createDataset reviews wordToIndex =
  let texts       = map (\r -> B.pack (encode (text r))) reviews
      ratings     = map (rating) reviews
      wordIndices = map (map wordToIndex . preprocess) texts
  in zip (map (asTensor) wordIndices)
        (map asTensor ratings)


createDataloader ::
  [(Tensor, Tensor)] -> -- pairs of (input, label)
  Int ->               -- batch size
  [(Tensor, Tensor)]   -- dataloader
createDataloader pairs batchSize =
  let batches      = chunksOf batchSize pairs
      fullBatches  = filter ((== batchSize) . length) batches
      makeBatch grp =
        let (xs, ys) = unzip grp
            x        = stack (Dim 0) xs
            y        = stack (Dim 0) ys
        in (x, y)
  in map makeBatch fullBatches

-- découpe une liste en sous‐listes de taille n (la dernière peut être plus petite)
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs =
  let (f, r) = splitAt n xs
  in f : chunksOf n r




allowedChars :: [Char]
allowedChars = "abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüæœ "

isAllowedChar :: Word8 -> Bool
isAllowedChar w =
  let c = head (encode [toLower (toEnum (fromEnum w))])
  in c `elem` (map (head . encode . (:[])) allowedChars)



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



