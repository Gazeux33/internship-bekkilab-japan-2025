module Main (main) where

import Torch
import qualified Torch.Functional.Internal as FI
import qualified Torch.Functional as F

trainingData :: [([Float],Float)]
trainingData = [([1,1],0),([1,0],1),([0,1],1),([0,0],0)]

type LinearLayer = (Tensor, Tensor)


data Layer = LinearL LinearLayer | Tanh | Sigmoid deriving (Show)

initLinearLayer :: Int -> Int -> LinearLayer
initLinearLayer i o = (zeros' [i,o], zeros' [o])

forwardLinear :: LinearLayer -> Tensor -> Tensor
forwardLinear (w, b) x = FI.matmul x w + b

forwardLayer :: Layer -> Tensor -> Tensor
forwardLayer (LinearL ll) x = forwardLinear ll x
forwardLayer Tanh       x = F.tanh x
forwardLayer Sigmoid    x = F.sigmoid x

sampleLayer :: Layer -> IO Layer
sampleLayer (LinearL (w,b)) = do
  w' <- randIO' (shape w)
  b' <- randIO' (shape b)
  return $ LinearL (w', b')
sampleLayer act = return act

sampleModel :: [Layer] -> IO [Layer]
sampleModel = mapM sampleLayer

forwardModel :: [Layer] -> Tensor -> Tensor
forwardModel ls x0 = foldl (\x l -> forwardLayer l x) x0 ls




computeGradient :: Layer -> Tensor -> Tensor -> (Layer, Tensor)
computeGradient (LinearL (w, b)) input grad = 
    let gradW = FI.matmul (FI.transpose input) grad
        gradB = FI.sumDim 0 False grad
        gradInput = FI.matmul grad (FI.transpose w)
    in (LinearL (gradW, gradB), gradInput)
computeGradient Tanh input grad =
    let tanhGrad = 1 - (F.tanh input) * (F.tanh input)
        gradInput = grad * tanhGrad
    in (Tanh, gradInput)
computeGradient Sigmoid input grad =
    let sigmoidX = F.sigmoid input
        sigmoidGrad = sigmoidX * (1 - sigmoidX)
        gradInput = grad * sigmoidGrad
    in (Sigmoid, gradInput)


-- updateModel :: Model -> Tensor -> Float -> Model
-- updateModel model loss lr = 

mse :: Tensor -> Tensor -> Tensor
mse actual predicted = 
    let diff = sub predicted actual
        squared = diff * diff
    in FI.meanAll squared Float

main :: IO ()
main = do
  
  let model =
        [ LinearL (initLinearLayer 1 2)
        , Tanh
        , LinearL (initLinearLayer 2 3)
        , Sigmoid
        ]
  initializedModel <- sampleModel model

  x_example <- randIO' [4,1]
  let y_predicted = forwardModel model x_example
  putStrLn $ "shape of output: " ++ show (shape y_predicted)
  y_example <- randIO' [4,3]

  let error = mse y_example y_predicted
  putStrLn $ "error: " ++ show (asValue error :: Float)


  