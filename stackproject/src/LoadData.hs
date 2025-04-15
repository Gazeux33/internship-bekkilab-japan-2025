module LoadData
    ( DataRow(..)
    , loadCSV
    , dataToTensors
    , createDataloader
    , createDataset
    , Dataloader
    , MyDataset
    , Batch
    ) where


import Torch
import qualified Data.ByteString.Lazy as BL
import qualified Torch.Functional.Internal as FI
import qualified Data.Csv as Csv
import qualified Data.Vector as V
import Data.Time (UTCTime) 
import Data.Time.Format (parseTimeM, defaultTimeLocale)
import Data.ByteString.Char8 (unpack)

type Dataloader = [(Tensor, Tensor)]
type MyDataset = [(Tensor, Tensor)]
type Batch = (Tensor, Tensor)

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
createInput tensor winSize =
    [ Torch.squeezeAll $ FI.slice tensor 0 i (i + winSize) 1 | i <- [0 .. Torch.size 0 tensor - winSize] ]


createTarget :: Tensor -> Int -> [Tensor]
createTarget tensor winSize = [ FI.slice tensor 0 (i + winSize) (i + winSize + 1) 1 | i <- [0 .. Torch.size 0 tensor - winSize - 1] ]


createDataset :: Tensor -> Int -> MyDataset
createDataset tensor winSize = zip (createInput tensor winSize) (createTarget tensor winSize)


chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = let (chunk, rest) = splitAt n xs in chunk : chunksOf n rest


stackTensors :: [Tensor] -> Tensor
stackTensors ts = Torch.stack (Dim 0) ts



createDataloader :: MyDataset -> Int -> Dataloader
createDataloader dataset batchSize =
    map processBatch (filter (\chunk -> length chunk == batchSize) (chunksOf batchSize dataset))
    where
        processBatch batch =
            let inputs = map fst batch  
                targets = map snd batch 
                inputBatch = stackTensors inputs 
                targetBatch = Torch.squeezeDim 1 (stackTensors targets)
            in (inputBatch, targetBatch)
            

dataToTensors :: V.Vector DataRow -> Tensor
dataToTensors records = 
    let features = map (\r -> [realToFrac (meanTemp r) :: Float]) (V.toList records)
    in asTensor (features :: [[Float]])

