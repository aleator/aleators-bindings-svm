{-# LANGUAGE ForeignFunctionInterface, BangPatterns, ScopedTypeVariables, TupleSections 
   #-}
module Main where

import qualified Data.Vector.Storable as V
import Data.Vector.Storable ((!))
import Bindings.SVM
import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.ForeignPtr
import qualified Foreign.Concurrent as C
import Foreign.Marshal.Utils
import Control.Applicative
import System.IO.Unsafe
import Foreign.Storable
import Control.Monad


{-# SPECIALIZE convertDense :: V.Vector Double -> V.Vector C'svm_node #-}
{-# SPECIALIZE convertDense :: V.Vector Float -> V.Vector C'svm_node #-}
convertDense :: (V.Storable a, Real a) => V.Vector a -> V.Vector C'svm_node
convertDense v = V.generate (dim+1) readVal
    where
        dim = V.length v
        readVal !n | n >= dim = C'svm_node (-1) 0
        readVal !n = C'svm_node (fromIntegral n) (realToFrac $ v ! n)


withProblem :: [(Double, V.Vector Double)] -> (Ptr C'svm_problem -> IO b) -> IO b
withProblem v op = -- Well. This turned out super ugly. Possibly even wrong.
                   V.unsafeWith xs $ \ptr_xs ->
                   V.unsafeWith y  $ \ptr_y -> 
                    let optrs = offsetPtrs ptr_xs
                    in V.unsafeWith optrs $ \ptr_offsets ->
                        with (C'svm_problem (fromIntegral dim) ptr_y ptr_offsets) op
    where 
        dim = length v
        lengths = map (V.length . snd) v
        offsetPtrs addr = V.fromList 
                          [addr `plusPtr` (idx * sizeOf (xs ! 0))
                          | idx <- scanl (+) 0 lengths]
        y   = V.fromList . map (realToFrac . fst) $ v
        xs  = V.concat . map (extractSvmNode.snd) $ v
        extractSvmNode x = convertDense $ V.generate (V.length x) (x !)

newtype SVM = SVM (ForeignPtr C'svm_model)

modelFinalizer :: Ptr C'svm_model -> IO ()
modelFinalizer = \modelPtr -> do
    with modelPtr (c'svm_free_and_destroy_model)

loadSVM :: FilePath -> IO SVM
loadSVM fp = do
    ptr <- withCString fp c'svm_load_model
    let fin = modelFinalizer ptr
    SVM <$> C.newForeignPtr ptr fin


predict :: SVM -> V.Vector Double -> Double
predict (SVM fptr) vec = unsafePerformIO $
                           withForeignPtr fptr $ \modelPtr -> 
                           let nodes = convertDense vec
                           in realToFrac <$> V.unsafeWith nodes (c'svm_predict modelPtr)

dummyParameters = C'svm_parameter {
      c'svm_parameter'svm_type = c'LINEAR
    , c'svm_parameter'kernel_type = c'RBF
    , c'svm_parameter'degree = 1
    , c'svm_parameter'gamma  = 0.01
    , c'svm_parameter'coef0  = 0.1
    , c'svm_parameter'cache_size = 10
    , c'svm_parameter'eps = 0.00001
    , c'svm_parameter'C   = 0.01
    , c'svm_parameter'nr_weight = 0
    , c'svm_parameter'weight_label = nullPtr
    , c'svm_parameter'weight       = nullPtr
    , c'svm_parameter'nu = 0.1
    , c'svm_parameter'p  = 0.1
    , c'svm_parameter'shrinking = 0
    , c'svm_parameter'probability = 0
                    }

foreign import ccall "wrapper"
  wrapPrintF :: (CString -> IO ()) -> IO (FunPtr (CString -> IO ()))


trainSVM :: [(Double, V.Vector Double)] -> IO SVM
trainSVM dataSet = do
    pf <- wrapPrintF (\cstr -> peekCString cstr >>= print . (, ":HS"))
    c'svm_set_print_string_function pf
    modelPtr <- withProblem dataSet $ \ptr_problem ->
                with dummyParameters $ \ptr_parameters ->
                    c'svm_train ptr_problem ptr_parameters
    SVM <$> C.newForeignPtr modelPtr (modelFinalizer modelPtr) 

main = do
    --mPtr <- withCString "model" c'svm_load_model
    svm <- loadSVM "model"
    let positiveSample = V.fromList 
                  [0.708333, 1, 1, -0.320755, -0.105023, -1
                  , 1, -0.419847, -1, -0.225806, 1, -1]
        negativeSample = V.fromList
                  [0.583333 ,-1 ,0.333333 ,-0.603774 ,1 ,-1
                  ,1 ,0.358779 ,-1 ,-0.483871 ,-1 ,1]

    let 
        pos = predict svm positiveSample
        neg = predict svm negativeSample 
    print (pos,neg)
    print "Training"
    let trainingData = [(-1, V.fromList [0])
                       ,(-1, V.fromList [20])
                       ,(1, V.fromList [21])
                       ,(1, V.fromList [50])
                        ]
    svm2 <- trainSVM trainingData
    print $ predict svm2 $ V.fromList [0]
    print $ predict svm2 $ V.fromList [5]
    print $ predict svm2 $ V.fromList [12]
    print $ predict svm2 $ V.fromList [40]


