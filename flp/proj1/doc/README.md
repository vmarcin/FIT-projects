Project: plg-2-nka
Author: xmarci10

TIN study text: https://wis.fit.vutbr.cz/FIT/st/cfs.php.cs?file=%2Fcourse%2FTIN-IT%2Ftexts%2Fopora.pdf&cid=13508
'*' symbol is used for items that go beyond the project assignment

Arguments parsing:
    - see ParseArgs module
    * the program supports multiple arguments at the time [-i, -1, -2] 
      (addition to the assignment)
    - in the case that more arguments are given the program prints all outputs 
      in the order given by the above-mentioned list

Input grammar parsing (only right linear grammar is accepted): -i option
    - see ParseGrammar module
    - this module contains all functions used to check the correctness of an 
      input grammar and also a definition of data type which is used to 
      represents an input grammar
    - in case of a mistake in an input grammar, the program prints an error 
      message that describes the mistake in more detail
    - given input must contain at least three lines: 
        
        1. NONTERMS
        2. terms
        3. START SYMBOL
        4. either first rule or nothing as the rules set could be empty

    - all empty lines at the end of the input file and empty lines between 
      the rules are ignored
    * the support of simple rules is also implemented in addition to the 
      assignment

Linear grammar 2 regular grammar transformation: -1 option
    - see TransformGrammar module
    - input linear grammar is transformed into regular grammar using the 
      algorithm in the TIN study text (see sentence 3.2)

Regular grammar 2 finite automata transformation: -2 option
    - see FiniteAutomata module
    - a regular grammar is transformed into a finite automata using the 
      algorithm in the TIN study text (see sentence 3.6)

OUTPUT NOTES:
    - if the resulting grammar or automata does not contain any of the defined 
      parts (e.g. no final states or no rules/transitions), an empty line is in 
      the output instead
    - all parts of output are always sorted by 'sort' function (tems, nonterms, 
      rules, states, ...)