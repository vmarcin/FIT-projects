{-# LANGUAGE BangPatterns #-}

module ParseGrammar where

import Data.List
import Data.List.Split(splitOn)
import Data.Char

-- type of grammar's nonterminals
type NonTerminal = String

-- type of grammar's terminals
type Terminal = String

-- type of grammar's rules
type Rule = (String, String)

-- rlg represents a type of right linear grammar
data RLG = RLG { nonterminals :: [NonTerminal] 
               , alphabet :: [Terminal] 
               , rules :: [Rule] 
               , start :: NonTerminal } 

-- show function for righ linear grammar
instance Show RLG where
    show (RLG n a r s) =
        intercalate "," n ++ "\n" ++
        intercalate "," a ++ "\n" ++ s ++ "\n" ++
        intercalate "\n" (map (\rule -> fst rule ++ "->" ++ snd rule) r) 

isNonTerminal :: Char -> Bool
isNonTerminal n = n `elem` ['A'..'Z']

isTerminal :: Char -> Bool
isTerminal t = t `elem` ['a'..'z'] 

-- Syntactic control of nonterminals. 'nonterms' is the first line of an input file.
checkNonTerminals :: String -> [String]
checkNonTerminals [] = []
checkNonTerminals nonterms =  
    if (head nonterms /= ',') && not (",," `isInfixOf` nonterms) &&
        (last nonterms /= ',') && all (\ n -> isNonTerminal n || n == ',') nonterms
        then splitOn "," nonterms 
        else error "[SYNTAX ERROR] Wrong format nonterminals!"

-- Syntactic control of nonterminals. 'terms' is the second line of an input file.
checkTerminals :: String -> [String]
checkTerminals [] = []
checkTerminals terms =
    if (head terms /= ',') && not (",," `isInfixOf` terms) &&
        (last terms /= ',') && all (\ n -> isTerminal n || n == ',') terms
        then splitOn "," terms 
        else error "[SYNTAX ERROR] Wrong format terminals!"

-- Checks start symbol's syntax/semantics and returns a start symbol or reports an error
checkStartSymbol :: [String] -> String -> String
checkStartSymbol nonterms start =
    case start of
        [a]
         | start `elem` nonterms -> start
         | not $ isNonTerminal a -> error "[SYNTAX ERROR] Wrong start symbol!"
         | otherwise -> error "[SEMANTIC ERROR] Wrong start symbol!"
        [] -> error "[SYNTAX ERROR] Empty start symbol!"
        _ -> error "[SYNTAX ERROR] Wrong start symbol!"

-- Checks syntax and returns rule (converted into internal representation) or 
-- report error in case of wrong syntax 
checkRuleSyntax :: String -> Rule
checkRuleSyntax rule =
   case rule of
        (left:'-':'>':right) 
            | isNonTerminal left && noeps == "" && right /= [] 
            -> ([left], "#")
            | isNonTerminal left && right /= "" && all isTerminal noeps
            -> ([left], noeps)
            | isNonTerminal left && right /= "" && isNonTerminal (last noeps) && 
                all isTerminal (init noeps)
            -> ([left], noeps)
            | otherwise
            -> error "[SYNTAX ERROR] Wrong rule's syntax!"
            -- remove epsilons as they have no effects
            where noeps = filter (/='#') right
        _ -> error "[SYNTAX ERROR] Wrong rule's syntax!"

-- Checks if all symbols used in grammar belongs to (terms `union` nonterms `union` "#")
checkRulesSemantics :: [Rule] -> [String] -> [String] -> Bool
checkRulesSemantics rules nonterms terms =
    let symbols = nonterms ++ terms ++ ["#"] in
    all (\rule -> 
            all (`elem` symbols) (map (: []) (fst rule)) && 
            all (`elem` symbols) (map (: []) (snd rule))) rules 

-- Checks rule's semantics and return rules or report error when sth worng
checkRules :: [String] -> [String] -> [String] -> [Rule]
checkRules rules nonterms terms = 
    let r = map checkRuleSyntax rules in 
    if checkRulesSemantics r nonterms terms 
        then r 
        else error "[SEMANTIC ERROR] Wrong rule's semantic!"

-- Function separates an input file into nonterms, terms, start symbol and rules,
-- checks every part and also force the evaluation because we want to know if
-- input is valid before we start transformation into desired form. Also all of
-- loaded parts are after check sorted and converted into internal represenation.
loadGrammar :: [String] -> RLG
loadGrammar input = 
    case input of
        (n:t:s:p) ->
            let ! nonterms = checkNonTerminals n in
            let ! terms = checkTerminals t in
            -- if p (which represents list of rules) is empty or contains only
            -- empty lines it means that we have no rules and simply set rules to []
            -- if p is not empty we check rules but before that we filter out empty
            -- lines as we don't consider them as mistakes 
            let ! rules = if all null p 
                            then [] 
                            else checkRules (filter (not.null) p) nonterms terms 
            in
            let ! start = checkStartSymbol nonterms s in 
            RLG (sort $ nub nonterms) 
                (sort $ nub terms) 
                (sort $ nub rules) 
                start
        _ -> error "[SYNTAX ERROR] Wrong input format!"