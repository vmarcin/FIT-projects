module Main (main) where
    
import Control.Monad
import System.Environment
import System.IO 
import Data.List
import Debug.Trace

-- own modules
import ParseArgs
import ParseGrammar
import TransformGrammar
import FiniteAutomata

main :: IO ()
main = do
    args <- getArgs
    let (options@[opt1, opt2, opt3], source) = parseArgs args
    input <- fmap lines $ if source /= [] then readFile source else getContents
    
    let grammar = loadGrammar input
    let rlg = transformGrammar grammar 
    let fa = rrg2fa rlg

    when opt1 $ print grammar
    when opt2 $ print rlg
    when opt3 $ print fa