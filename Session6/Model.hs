{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Model where 

import qualified Torch.Functional as F
import qualified Torch.Functional.Internal as FI
import qualified Data.ByteString.Lazy as B 
import Control.Monad (foldM,when)
import GHC.Generics
import Torch
import Embedding
import MLP
import ML.Exp.Chart   (drawLearningCurve)

import Evaluation

data Model = Model {
    embeddings :: Embedding,
    mlp :: MLP
  } deriving (Show, Generic, Parameterized)


modelForward :: Model -> Tensor -> Tensor
modelForward Model {..} x = mlpForward mlp $ embeddingForward embeddings x


computeCrossEntropyLoss :: Tensor -> Tensor -> Tensor
computeCrossEntropyLoss output target = 
  -- output : (B*T, vocab_size)
  -- target : (B*T)
    let
      weight = ones' [last (shape output)]
      loss = FI.cross_entropy_loss output target weight 1 (-100) 0.0

    in
      loss


saveEmnding :: Model -> FilePath -> Bool -> Int -> IO ()
saveEmnding Model {..} path verbose iter  = do
  let p = "data/embedding/models/embedding_model_" ++ show iter ++ ".params"
  saveParams embeddings p
  if verbose then
    putStrLn $ "Model saved at: " ++ p
  else
    pure ()



processBatch :: Optimizer o => Model -> o -> Float -> (Tensor, Tensor) -> IO (Model, Float)
processBatch model optimizer lr (input, label) = do
  let output = modelForward model input
  let loss = computeCrossEntropyLoss output label 
  (newModel, _) <- runStep model optimizer loss (realToFrac lr)  
  pure (newModel, asValue loss)

processTrainEpoch :: Optimizer o => 
                    [(Tensor, Tensor)] -- Training data
                    -> Model -- Initial model
                     -> [[String]]           -- Eval data
                    -> (B.ByteString -> Int) -- Word to index function
                    -> o -- Optimizer
                    -> Float -- Learning rate
                    -> Int -- Epoch
                    -> Int -- Print frequency
                    -> Int  -- Save frequency
                    -> Int -- eval frequency
                    -> FilePath -- Save path
                    -> IO (Model, [Float], [Float])  -- model, list of batch losses, and list of eval scores
processTrainEpoch dataloader model evalData wordToIndex optimizer lr epoch printFreq saveFreq evalFreq savePath   = do
    foldM (\(m', losses, evalScores) (i, batch) -> do
                (newModel, loss) <- processBatch m' optimizer lr batch
                let newLosses = loss:losses

                -- print and draw curve at intervals
                when (i `mod` printFreq == 0) $ do
                    putStrLn $ "Epoch " ++ show epoch ++ " | Batch " ++ show i ++ " | Loss: " ++ show loss
                    drawLearningCurve "Session6/output/word2vec-training.png" "Word2Vec Training Losses Curve" [("Training Loss", reverse newLosses)]
                
                -- save model periodically
                when (i `mod` saveFreq == 0) $
                    saveEmnding newModel savePath True i

                -- evaluate model periodically and track scores
                if (i `mod` evalFreq == 0)
                   then do
                     score <- evalModel evalData (embeddings newModel) wordToIndex
                     putStrLn $ "Evaluation Score: " ++ show score
                     let newEvalScores = (fromIntegral score):evalScores
                     
                     drawLearningCurve "Session6/output/word2vec-eval.png" "Word2Vec Evaluation Metrics" 
                                     [("Evaluation Score", reverse newEvalScores)]
                     pure (newModel, newLosses, newEvalScores)
                   else
                     pure (newModel, newLosses, evalScores)
            ) (model, [], []) (zip [1..] dataloader)

trainWithLossTracking :: Optimizer o => 
                       [(Tensor, Tensor)]   -- Training data (batches)
                    -> Model                  -- Initial model
                    -> [[String]]           -- Eval data
                    -> (B.ByteString -> Int) -- Word to index function
                    -> o                    -- Optimizer
                    -> Float                -- Learning rate
                    -> Int                  -- Number of epochs
                    -> Int                  -- Print frequency
                    -> Int  -- Save frequency
                    -> Int  -- eval frequency
                    -> FilePath -- Save path
                    -> IO (Model, [Float], [Float]) -- final model, list of all batch losses, and list of all eval scores
trainWithLossTracking trainData initialModel evalData wordToIndex optimizer lr epochs printFreq saveFreq evalFreq savePath = do
    foldM (\(model, trainLosses, evalScores) epoch -> do
              (newModel, epochLosses, epochEvalScores) <- processTrainEpoch trainData model evalData wordToIndex optimizer lr epoch printFreq saveFreq evalFreq savePath

              let allLosses = trainLosses ++ epochLosses
              let allEvalScores = evalScores ++ epochEvalScores

              pure (newModel, allLosses, allEvalScores)
          ) (initialModel, [], []) [1..epochs]


