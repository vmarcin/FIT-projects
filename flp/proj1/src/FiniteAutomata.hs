module FiniteAutomata where

import Data.List
import Data.Maybe

-- own modules
import ParseGrammar

data FA = FA { states :: [String]
             , startState :: String 
             , finalStates :: [String]
             , rulesFa :: [(String, String, String)] }

instance Show FA where
    show (FA q s f p) =
        intercalate "," q ++ "\n" ++
        s ++ "\n" ++
        intercalate "," f ++ "\n" ++ 
        intercalate "\n" (map (\(a,b,c) -> a ++ "," ++ b ++ "," ++ c) p) 

isNotEpsilonRule :: Rule -> Bool
isNotEpsilonRule (left, right) = right /= "#" 

-- for each nonterm create a state represented by number
-- first nonterm in nonterm list is represented by 0, second by 1, and so on...
nonterms2states :: [NonTerminal] -> Integer -> [String]
nonterms2states nonterms state =
    case nonterms of
        [] -> [] 
        (x:xs) -> show state : nonterms2states xs (state + 1)

getStateByNonterm :: [(String, String)] -> NonTerminal -> String
getStateByNonterm tuples state = fromMaybe "" (lookup state tuples)

-- returns set of final nonterms (these which has '#' on the right side)
finalNontermsSet :: [Rule] -> [NonTerminal]
finalNontermsSet [] = []
finalNontermsSet (x:xs) =
    case x of
        (a,['#']) -> a : finalNontermsSet xs
        _ -> finalNontermsSet xs

-- finds coresponding states to final nonterms a returns them
finalStatesSet :: [(String, String)] -> [NonTerminal] -> [String]
finalStatesSet tuples = map (getStateByNonterm tuples) 

-- transforms grammar rule into FA transition format by spliting righ side of the rule
rlgRule2faFormat :: Rule -> (String, String, String)
rlgRule2faFormat (left, right) = (left, [head right], tail right)

-- convert each RRG rule (except epsilon rules) to FA transition
rlgRules2faRules :: [(String, String)] -> [Rule] -> [(String, String, String)]
rlgRules2faRules tuples rules = 
    -- filter out epsilon rules as they don't represent any transition in FA
    let rulesWithoutEps = filter isNotEpsilonRule rules in
    map (\rule -> 
            let (nonterm1, symbol, nonterm2) = rlgRule2faFormat rule in
            (getStateByNonterm tuples nonterm1, symbol, getStateByNonterm tuples nonterm2)
        ) rulesWithoutEps

-- converts right regular grammar into finite automata
rrg2fa :: RLG -> FA
rrg2fa grammar =
    let faStates = nonterms2states (nonterminals grammar) 0 in
    -- make tuples which tell us which nonterm belongs to which state
    let tuples = zip (nonterminals grammar) faStates in
    let faStartState = getStateByNonterm tuples $ start grammar in
    let faFinalStates = finalStatesSet tuples (finalNontermsSet (rules grammar)) in
    let faRules = rlgRules2faRules tuples (rules grammar) in

    FA faStates faStartState (sort faFinalStates) faRules