module Data (
    loadData
) where

import Torch
import qualified Data.Vector as V
import qualified Data.Csv as Csv
import qualified Data.ByteString.Lazy as BL




    --  #   Column             Non-Null Count  Dtype  
-- ---  ------             --------------  -----  
--  0   Serial No.         400 non-null    int64  
--  1   GRE Score          400 non-null    int64  
--  2   TOEFL Score        400 non-null    int64  
--  3   University Rating  400 non-null    int64  
--  4   SOP                400 non-null    float64
--  5   LOR                400 non-null    float64
--  6   CGPA               400 non-null    float64
--  7   Research           400 non-null    int64  
--  8   Chance of Admit    400 non-null    float64
data DataRow = DataRow {
    id :: Int,
    gre :: Int,
    toefl :: Int,
    universityRating :: Int,
    sop :: Float,
    lor :: Float,
    cgpa :: Float,
    research :: Int,
    admitChance :: Float
} deriving (Show)


instance Csv.FromRecord DataRow where
    parseRecord v
        | V.length v >= 9 = DataRow
            <$> Csv.parseField (v V.! 0)  
            <*> Csv.parseField (v V.! 1)  
            <*> Csv.parseField (v V.! 2)  
            <*> Csv.parseField (v V.! 3)   
            <*> Csv.parseField (v V.! 4)  
            <*> Csv.parseField (v V.! 5)  
            <*> Csv.parseField (v V.! 6)  
            <*> Csv.parseField (v V.! 7) 
            <*> Csv.parseField (v V.! 8) 
        | otherwise = fail $ "Error , line legth " ++ show (V.length v)


loadCSV :: FilePath -> IO (Either String (V.Vector DataRow))
loadCSV filePath = do
    csvData <- BL.readFile filePath
    return $ Csv.decode Csv.HasHeader csvData


loadData :: String -> IO (Either String ([Float], [Float]))
loadData filePath = do
    csvResult <- loadCSV filePath
    case csvResult of
        Left err -> return (Left err)
        Right rows -> do
            let vector = dataToTensors rows
            return (Right vector)


-- X = GRE Score        Y = TOEFL Score
dataToTensors :: V.Vector DataRow -> ([Float], [Float])
dataToTensors records = 
    let features = map (\r -> fromIntegral (gre r)) (V.toList records)
        labels = map (\r -> fromIntegral (toefl r)) (V.toList records)
    in (features, labels)