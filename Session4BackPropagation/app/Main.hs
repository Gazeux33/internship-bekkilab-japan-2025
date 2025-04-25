module Main (main) where

import Lib
import Torch
import qualified Torch.Functional.Internal as FI
import qualified Torch.Functional as F

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

main :: IO ()
main = do
  
  let model =
        [ LinearL (initLinearLayer 1 2)
        , Tanh
        , LinearL (initLinearLayer 2 3)
        , Sigmoid
        ]

  initializedModel <- sampleModel model
  putStrLn $ map (show shape) model

  inp <- randIO' [4,1]
  let out = forwardModel initializedModel inp
  putStrLn $ "shape of output: " ++ show (shape out)
  putStrLn "Hello, Haskell!"