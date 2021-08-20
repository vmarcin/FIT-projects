module TransformGrammar (transformGrammar) where

import Data.List

-- own modules
import ParseGrammar

-- checks if 'r' is simple rule it means that it has a format A->B : A,B <- nonterms
isSimpleRule :: Rule -> Bool
isSimpleRule r =
    case r of
        (_, [b]) -> isNonTerminal b
        _ -> False

-- checks if rule has a format A->aB v A-># v A->a : A,B <- nonterms, a <- terms
isRegularRule :: Rule -> Bool
isRegularRule r = 
    case r of
        (_, [b])  
            | b == '#' || isTerminal b -> True
            | otherwise -> False
        (_, [b, c])   
            | isTerminal b && isNonTerminal c -> True
            | otherwise -> False
        _ -> False 

-- gets right linear rule (left,right) and generates new right regular rules 
-- which are equivalent 
generateRules :: [Rule] -> String -> String -> Int -> [Rule]        
generateRules rules left right i =
    let newNonterm = head left : show i in
    let newRight = head right : newNonterm in
    if isRegularRule (left, right) 
        then if isTerminal $ last right 
                then rules ++ [(left, newRight), (newNonterm, "#")]
                else rules ++ [(left, right)]
        else generateRules (rules ++ [(left, newRight)]) newNonterm (tail right) (i+1)     

-- gets right linear rules and integer which serves as counter of newly added nonterms
-- e.g. nonterms created from rule A->aaB will be A0 and A1 
replaceRules :: [Rule] -> Int -> [Rule]
replaceRules rules i = 
    case rules of 
        [] -> []
        (x : xs) -> 
            let r = uncurry (generateRules []) x i in
            r ++ replaceRules xs (i + length r - 1) 

-- right linear rules with same nonterminal on the left side are given and 
-- they are transformed into right regular rules
transformRules :: [Rule] -> [Rule]
transformRules rules = replaceRules rules 0

-- filters all nonterminals from given rules
filterNonTerminals :: [Rule] -> [NonTerminal]
filterNonTerminals rules = [fst rule | rule <- rules]

-- gets list 'nonterm' with one nonterm and computes a set of all 
-- reacheable nonterms by using simple rules (TIN script algorithm 4.5)
simplePath :: [NonTerminal] -> [Rule] -> [NonTerminal] -> [NonTerminal]
simplePath [] _ _ = []
simplePath nonterms rules seen =
    let ni = foldr (\n acc -> 
                        let r = filter (\rule -> fst rule == n) rules in
                        acc ++ [snd rule | rule <- r]
                    ) [] nonterms in
    nonterms ++ simplePath (ni \\ seen) rules (seen ++ ni)

-- replaces simple rules according to sentence 3.2 (4)
removeSimpleRules :: [Rule] -> [Rule] -> [Rule]
removeSimpleRules simpleRules noSimpleRules =
    -- filter all nonterminals on the left side of simple rules
    let nonterms = nub $ filterNonTerminals simpleRules in 
    foldr (\n acc -> 
                -- sentence 3.2 (4a)
                let nA = tail $ nub $ simplePath [n] simpleRules [n] in
                -- sentence 3.2 (4b)
                let r = filter (\rule -> fst rule `elem` nA) noSimpleRules in
                acc ++ [(n,b) | b <- [snd rule | rule <- r]] 
            ) 
        [] nonterms

-- transform right linear grammar into right regular grammar
transformGrammar :: RLG -> RLG
transformGrammar grammar =
    let n = nonterminals grammar in 

    -- filter just non simple rules
    let rulesWithoutSimple = filter (not . isSimpleRule) $ rules grammar in
    -- create a list of lists where in each of them are just rules with same 
    -- nonterminal on the left side e.g. [[A->aB, A->aa],[B->b],...]
    let groupedRules = map (\n -> filter (\rule -> fst rule == n) rulesWithoutSimple) n in
    -- transform rulesWithoutSimple into regular grammar rules format
    let rlgRules = concatMap transformRules groupedRules in  
    
    -- filter just simple rules
    let simpleRules = filter isSimpleRule $ rules grammar in
    -- simple rules removed and newly created rules added to rlgRules 
    let finalRules = rlgRules ++ removeSimpleRules simpleRules rlgRules in

    -- due to the transformation of rules new nonterminals could arise so we
    -- get them from newly created rules
    let rlgNonterms = nub (filterNonTerminals rlgRules) in
    
    -- transformed grammar is returned
    RLG (sort $ n `union` rlgNonterms) 
        (alphabet grammar) 
        (sort finalRules) 
        (start grammar)  