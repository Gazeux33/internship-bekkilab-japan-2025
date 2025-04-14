module LoadData
    ( DataRow(..)
    , loadCSV
    , dataToTensors
    , createDataloader
    , createDataset
    ) where


import Torch
import qualified Data.ByteString.Lazy as BL
import qualified Torch.Functional.Internal as FI
import qualified Data.Csv as Csv
import qualified Data.Vector as V
import Data.Time (UTCTime) 
import Data.Time.Format (parseTimeM, defaultTimeLocale)
import Data.ByteString.Char8 (unpack)

instance Csv.FromField UTCTime where
    parseField bs = case parseTimeM True defaultTimeLocale "%Y/%-m/%-d" (unpack bs) of
                        Just t -> pure t
                        Nothing -> fail "Could not parse date"


data DataRow = DataRow {
    date :: UTCTime,
    meanTemp :: Float
} deriving (Show)


instance Csv.FromRecord DataRow where
    parseRecord v
        | V.length v == 2 = DataRow
            <$> Csv.parseField (v V.! 0)
            <*> Csv.parseField (v V.! 1)
        | otherwise     = fail "Expected 2 fields for DataRow"


loadCSV :: FilePath -> IO (Either String (V.Vector DataRow))
loadCSV filePath = do
    csvData <- BL.readFile filePath
    return $ Csv.decode Csv.HasHeader csvData


createInput :: Tensor -> Int -> [Tensor]
createInput tensor winSize = [ FI.slice tensor 0 i (i + winSize) 1 | i <- [0 .. size 0 tensor - winSize] ]


createTarget :: Tensor -> Int -> [Tensor]
createTarget tensor winSize = [ FI.slice tensor 0 i (i + 1) 1 | i <- [winSize .. size 0 tensor - 1] ]


createDataset :: Tensor -> Int -> [(Tensor, Tensor)]
createDataset tensor winSize = zip (createInput tensor winSize) (createTarget tensor winSize)


createDataloader :: [(Tensor, Tensor)] -> Int -> [[(Tensor, Tensor)]]
createDataloader dataset batchSize = 
    chunks dataset
    where
        chunks [] = []
        chunks xs 
            | length xs >= batchSize = Prelude.take batchSize xs : chunks (drop batchSize xs)
            | otherwise = []
            

    
dataToTensors :: V.Vector DataRow -> Tensor
dataToTensors records = 
    let features = map (\r -> [realToFrac (meanTemp r) :: Float]) (V.toList records)
    in asTensor (features :: [[Float]])

