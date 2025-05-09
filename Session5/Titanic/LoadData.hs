module LoadData
    ( 
    loadCSV
    , dataToTensors
    , createDataloader


    ) where

import Torch
import Data.List.Split (chunksOf)
import qualified Data.ByteString.Lazy as BL
import qualified Torch.Functional.Internal as FI
import qualified Data.Csv as Csv
import qualified Data.Vector as V
 



-- PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

-- Preprocesssed -> Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
data DataRow = DataRow {
    survived :: Int,
    pClass :: Int,
    sex :: Int,
    age :: Int,
    sibSp :: Int,
    parch :: Int,
    fare :: Float,
    embarked :: Int
} deriving (Show)

instance Csv.FromRecord DataRow where
    parseRecord v
        | V.length v >= 8 = DataRow
            <$> Csv.parseField (v V.! 0)  -- Survived
            <*> Csv.parseField (v V.! 1)  -- Pclass
            <*> Csv.parseField (v V.! 2)  -- Sex (prétraité en Int)
            <*> Csv.parseField (v V.! 3)   -- Age
            <*> Csv.parseField (v V.! 4)  -- SibSp
            <*> Csv.parseField (v V.! 5)  -- Parch
            <*> Csv.parseField (v V.! 6)  -- Fare
            <*> Csv.parseField (v V.! 7)  -- Embarked (prétraité en Int)
        | otherwise = fail $ "Expected at least 8 fields for DataRow, got " ++ show (V.length v)


loadCSV :: FilePath -> IO (Either String (V.Vector DataRow))
loadCSV filePath = do
    csvData <- BL.readFile filePath
    return $ Csv.decode Csv.HasHeader csvData




dataToTensors :: V.Vector DataRow -> (Tensor, Tensor)
dataToTensors records = 
    let features = map (\r -> [
            fromIntegral (pClass r),
            fromIntegral (sex r),
            fromIntegral (age r),
            fromIntegral (sibSp r),
            fromIntegral (parch r),
            realToFrac (fare r),
            fromIntegral (embarked r)
            ]) (V.toList records)
        labels = map (\r -> [fromIntegral (survived r) :: Float]) (V.toList records)
    in (asTensor (features :: [[Float]]), asTensor (labels :: [[Float]]))


createDataloader :: (Tensor, Tensor) -> Int -> [(Tensor, Tensor)]
createDataloader (features, labels) batchSize =
    let numSamples = head (shape features)
        indices = [0..(numSamples - 1)]
        batches = filter (\batch -> length batch == batchSize) (chunksOf batchSize indices)
    in map (\idxs -> 
            let batchFeatures = indexSelect 0 (asTensor idxs) features
                batchLabels = indexSelect 0 (asTensor idxs) labels
            in (batchFeatures, batchLabels)
        ) batches
