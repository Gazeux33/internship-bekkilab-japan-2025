module Main (main) where

import Torch

import Lib

main :: IO ()
main = do
    someFunc
    let zeros = zeros' [10] :: Tensor
    print zeros
