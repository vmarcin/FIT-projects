import argparse 
import xml.etree.ElementTree as ET
import sys
import os
import re
import getopt

#return codes of interpret
ARGUMENT_ERROR = 10
FILE_ERROR_IN = 11
FILE_ERROR_OUT = 12
XML_ERR = 31
SYNLEX_ERR = 32
SEMANTIC_ERR = 52
WRONG_OPERAND_TYPE = 53
VAR_DOESNOT_EXIST = 54
FRAME_DOESNOT_EXIST = 55
MISSING_VALUE = 56
DIVISION_BY_ZERO = 57
WRONG_STRING_OPERATION = 58

#GLOBALS	
arg_t = None		#typ aktualne spracovavaneho operandu 
xml_inst = {}		#spracovany xml subor {instukcia: operandy}
inst_counter = 1 	#instruction pointer (cislo prave vykonavanej instrukcie)
call_stack = list()
labels = {}			#vsetky labely s ich poradovymi cislami v kode (instruction order)
frame_stack = list()
GF = {}				#globalny ramec
LF = None			#lokalny ramec
TF = None			#docastny ramec
data_stack = list() 
insts_e = 0			#priznak o pritomnosti prepinaca --insts 
vars_e = 0			#priznak o pritomnosti prepinaca --vars
stats_file = None	#vystupny subor so statistikami

#Funkcia skontroluje validnost operandu 't' typu 'type'
def check_type(t):
	if(t == None):
		print("ERROR: Syntax or lexical error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)
	if(re.match(r'^(int|bool|string)$', t)):
		return 
	else:
		print("ERROR: Syntax or lexical error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)

#Funkcia skontroluje validnost operandu 'label' typu 'label'
def check_label(label):
	if(label == None):
		print("ERROR: Syntax or lexical error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)
	if(re.match( r'^[a-zA-Z-_$&%*][a-zA-Z0-9-_$&%]*$', label)):
		return 
	else:
		print("ERROR: Syntax or lexical error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)

#Funkcia skontroluje validnost operandu 'var' typu 'var'
def check_var(var):
	if(var == None):
		print("ERROR: Syntax or lexical error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)
	if(re.match( r'^[LTG]F@[a-zA-Z-_$&%*][a-zA-Z0-9-_$&%*]*$', var)):
		return
	else:
		print("ERROR: Syntax or lexical error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)
	
#Funkcia skontroluje validnost symbolu 'symb'
def check_symb(symb):
	if(symb == None):
		symb = ''
	if(arg_t == 'int'):
		if(re.match( r'^(-|\+)?\d+$', symb) == None):
			print("ERROR: Syntax or lexical error!", file=sys.stderr)
			sys.exit(SYNLEX_ERR)
	elif(arg_t == 'bool'):
		if(re.match( r'^(true|false)$', symb) == None):	
			print("ERROR: Syntax or lexical error!", file=sys.stderr)
			sys.exit(SYNLEX_ERR)
	elif(arg_t == 'string'):
		if(re.match( r'^([^\s\x5C#]|\x5C[0-9]{3})*$', symb) == None):
			print("ERROR: Syntax or lexical error!", file=sys.stderr)
			sys.exit(SYNLEX_ERR)
	elif(arg_t == 'var'):
		if(re.match( r'^[LTG]F@[a-zA-Z-_$&%*][a-zA-Z0-9-_$&%*]*$', symb) == None):
			print("ERROR: Syntax or lexical error!", file=sys.stderr)
			sys.exit(SYNLEX_ERR)

#slovnik sluziaci na ulozenie informacii o jednotlivych instrukicach
instructions = {
	'MOVE': 		{1:check_var, 2:check_symb},
	'CREATEFRAME': 	{},
	'PUSHFRAME': 	{},
	'POPFRAME': 	{},
	'DEFVAR': 		{1:check_var},
	'CALL': 		{1:check_label},
	'RETURN': 		{},
	'PUSHS': 		{1:check_symb},
	'POPS': 		{1:check_var},
	'ADD': 			{1:check_var, 2:check_symb, 3:check_symb},
	'SUB': 			{1:check_var, 2:check_symb, 3:check_symb},
	'MUL': 			{1:check_var, 2:check_symb, 3:check_symb},
	'IDIV': 		{1:check_var, 2:check_symb, 3:check_symb},
	'LT':			{1:check_var, 2:check_symb, 3:check_symb},
	'GT':			{1:check_var, 2:check_symb, 3:check_symb},
	'EQ':			{1:check_var, 2:check_symb, 3:check_symb},
	'AND':			{1:check_var, 2:check_symb, 3:check_symb},
	'OR':			{1:check_var, 2:check_symb, 3:check_symb},
	'NOT':			{1:check_var, 2:check_symb},
	'INT2CHAR':		{1:check_var, 2:check_symb},
	'STRI2INT':		{1:check_var, 2:check_symb, 3:check_symb},
	'READ':			{1:check_var, 2:check_type},
	'WRITE':		{1:check_symb},
	'CONCAT':		{1:check_var, 2:check_symb, 3:check_symb},
	'STRLEN':		{1:check_var, 2:check_symb},
	'GETCHAR':		{1:check_var, 2:check_symb, 3:check_symb},
	'SETCHAR':		{1:check_var, 2:check_symb, 3:check_symb},
	'TYPE':			{1:check_var, 2:check_symb},
	'LABEL':		{1:check_label},
	'JUMP':			{1:check_label},
	'JUMPIFEQ':		{1:check_label, 2:check_symb, 3:check_symb},
	'JUMPIFNEQ':	{1:check_label, 2:check_symb, 3:check_symb},
	'DPRINT':		{1:check_symb},
	'BREAK':		{}
}

# funkcia nahradi vsetky escape sekvencie v retazci 'string' prislusnymi znakmi
# ktore vrati funkcia chr()
def remove_escape(string):
	new_string = None
	d = list() 
	d = [m.start() for m in re.finditer(r'\x5C[0-9]{3}', string)]
	x=0
	for index in d:
		char = chr(int(string[index+1-x:(index+4-x)]))
		string = string[:index-x] + char + string[(index-x+4):]	
		x+=3
	return string

# funkcia skontroluje ci existuje frame 'fr'
# ak existuje vrati ramec inak hlasi chybu
def check_frame(fr):
	frame = eval(fr)
	if(frame == None):
		print("ERROR: Frame does not exists!", file=sys.stderr)
		sys.exit(FRAME_DOESNOT_EXIST)
	else:
		return frame

# funkcia skontroluje ci existuje premenna 'var' 
# ak existuje vrati premennu inak hlasi chybu
def check_variable(var):
	variable = var.split("@")[1]
	if(eval(var.split("@")[0]).get(variable) == None):
		print("ERROR: Var does not exists!", file=sys.stderr)
		sys.exit(VAR_DOESNOT_EXIST)
	else:
		return variable

# funkcia skontroluje 'symbol' a vrati hodnotu a jeho typ
# v pripade ze symbol je premenna a nieje inicializovana funkcia hlasi chybu 
def check_symbol(symbol):
	if(symbol[0] == 'var'):
		frame = check_frame(symbol[1].split("@")[0])
		var = check_variable(symbol[1])
		if(frame[var][0] == None):
			print("ERROR: Missing value in variable!", file=sys.stderr)
			sys.exit(MISSING_VALUE)
		return {0:frame[var][0], 1:frame[var][1]}
	else:
		return {0:symbol[0], 1:symbol[1]} 

# skontroluje typ operandu a ak nie hlasi chybu
def check_operand_type(wanted_type, variable_type, is_var):
	if(variable_type != wanted_type):
		if(is_var[0] == 'var'):
			print("ERROR: Wrong type of operand!", file=sys.stderr)
			sys.exit(WRONG_OPERAND_TYPE)
		else:
			print("ERROR: Wrong type of operand!", file=sys.stderr)
			sys.exit(SEMANTIC_ERR)

###################################################################		
#	FUNKCIE PRE JEDNOTLIVE INSTRUKCIE   
###################################################################
def MOVE(arg_list):
	global inst_counter
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	symbol = check_symbol(arg_list[2]) #{0: 'string', 1:'ahoj'}
	frame_to[var_to] = symbol
	inst_counter+=1 

def CREATEFRAME(arg_list):
	global TF, inst_counter
	#vytvorenie noveho docasneho ramca
	TF = {}
	inst_counter+=1

def PUSHFRAME(arg_list):
	global TF, LF, inst_counter
	if(TF == None):
		print("ERROR: Frame does not exists!", file=sys.stderr)
		sys.exit(FRAME_DOESNOT_EXIST)
	#ak existuje docasny ramec dame ho na zasobnik ramcov
	#a stane sa s neho lokalny ramec
	frame_stack.append(TF)
	TF = None
	LF = frame_stack[-1]	
	inst_counter+=1

def POPFRAME(arg_list):
	global TF, LF, inst_counter
	if(LF == None):
		print("ERROR: Frame does not exists!", file=sys.stderr)
		sys.exit(FRAME_DOESNOT_EXIST)
	#Lokalny ramec sa stane docasnym
	TF = frame_stack.pop()
	#ak na zasobniku ramcov nic nieje lokalny ramec neexistuje
	if(frame_stack):
		LF = frame_stack[-1]
	else:
		LF = None
	inst_counter+=1

def DEFVAR(arg_list):
	global inst_counter
	frame = eval(arg_list[1][1].split("@")[0])
	var = arg_list[1][1].split("@")[1]
	if(frame == None):
		print("ERROR: Frame does not exists!", file=sys.stderr)
		sys.exit(FRAME_DOESNOT_EXIST)
	#vytvorenie novej neinicializovanej premennej
	frame[var] = {0: None, 1: None}
	inst_counter+=1

def CALL(arg_list):
	global inst_counter, call_stack
	#aktualna pozicia v kode sa ulozi na zasobnik volani
	call_stack.append((inst_counter+1))
	#skok na dany label
	if(labels.get(arg_list[1][1]) == None):
		print("ERROR: Label does not exists!", file=sys.stderr)
		sys.exit(SEMANTIC_ERR)
	else:
		inst_counter = labels.get(arg_list[1][1])

def RETURN(arg_list):
	global inst_counter	
	#navrat z funkcie zo zasobnika volani sa vyberie 
	#pozicia odkial bola volana prave ukoncena funkcia
	if(call_stack):
		inst_counter = call_stack.pop()
	else:
		print("ERROR: Missing value on call stack!", file=sys.stderr)
		sys.exit(MISSING_VALUE)

def PUSHS(arg_list):
	global inst_counter, data_stack
	symbol = check_symbol(arg_list[1])
	#ak sa jedna o validny symbol je pridany na datovy zasobnik
	data_stack.append(symbol)
	inst_counter+=1

def POPS(arg_list):
	global inst_counter
	#ak sa na zasobniku nieco nachadza do zadanej premennej
	#je ulozena hodnota z vrcholu zasobnika
	if(data_stack):
		frame_to = check_frame(arg_list[1][1].split("@")[0])
		var_to = check_variable(arg_list[1][1])
		frame_to[var_to] = data_stack.pop()
	else:
		print("ERROR: Missing value on data stack!", file=sys.stderr)
		sys.exit(MISSING_VALUE)
	inst_counter+=1

#funkcia ADD ma jeden argument naviac, ktory urcuje 
#typ operacie ktora sa ma vykonat naslene funkcie s 
#rovnakou syntaxou volaju funkciu ADD s typom operaciu ktoru chcu vykonat
def ADD(arg_list, operation = None):
	global inst_counter
	op2 = None
	op2_type = None

	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	symbol1 = check_symbol(arg_list[2])
	op1 = symbol1[1]
	op1_type = symbol1[0]

	#instrukcia not ma iba jeden argument
	if(operation != 'not'):
		symbol2 = check_symbol(arg_list[3])
		op2 = symbol2[1]
		op2_type = symbol2[0]
	
	if operation in [None, '-' , '*' , '/']:
		check_operand_type('int', op1_type, arg_list[2])
		check_operand_type('int', op2_type, arg_list[3])
	if operation in ['>', '<', '=']:
		check_operand_type(op1_type, op2_type, arg_list[3])
	if operation in ['and', 'or']:
		check_operand_type('bool', op1_type, arg_list[2])
		check_operand_type('bool', op2_type, arg_list[3])
	if(operation == 'not'):
		check_operand_type('bool', op1_type, arg_list[2])
	if(operation == 'concat'):
		check_operand_type('string', op1_type, arg_list[2])
		check_operand_type('string', op2_type, arg_list[3])

	if(operation == None):
		frame_to[var_to][0] = 'int'
		frame_to[var_to][1] = int(op1) + int(op2)
	elif(operation == '-'):
		frame_to[var_to][0] = 'int'
		frame_to[var_to][1] = int(op1) - int(op2)
	elif(operation == '*'):
		frame_to[var_to][0] = 'int'
		frame_to[var_to][1] = int(op1) * int(op2)
	elif(operation == '/'):
		if(int(op2) == 0):
			print("ERROR: Division by zero!", file=sys.stderr)
			sys.exit(DIVISION_BY_ZERO)
		frame_to[var_to][0] = 'int'
		frame_to[var_to][1] = int(op1) // int(op2)
	elif(operation == '<'):
		if(op1_type == 'int'):
			if(int(op1) < int(op2)):
				frame_to[var_to][1] = 'true'
			else:	
				frame_to[var_to][1] = 'false'
		elif(op1_type == 'bool'):
			if(op1 == 'false' and op2 == 'true'):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
		elif(op1_type == 'string'):
			if(op1 < op2):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
		frame_to[var_to][0] = 'bool'	
	elif(operation == '>'):
		if(op1_type == 'int'):
			if(int(op1) > int(op2)):
				frame_to[var_to][1] = 'true'
			else:	
				frame_to[var_to][1] = 'false'
		elif(op1_type == 'bool'):
			if(op1 == 'true' and op2 == 'false'):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
		elif(op1_type == 'string'):
			if(op1 > op2):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
		frame_to[var_to][0] = 'bool'	
	elif(operation == '='):
		if(op1_type == 'int'):
			if(int(op1) == int(op2)):
				frame_to[var_to][1] = 'true'
			else:	
				frame_to[var_to][1] = 'false'
		elif(op1_type == 'bool'):
			if(op1 == op2):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
		elif(op1_type == 'string'):
			if(op1 == op2):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
		frame_to[var_to][0] = 'bool'	
	elif(operation == 'and'):
		if(op1 == 'false' or op2 == 'false'):
			frame_to[var_to][1] = 'false'
		else:
			frame_to[var_to][1] = 'true'
		frame_to[var_to][0] = 'bool'
	elif(operation == 'or'):
		if(op1 == 'true' or op2 == 'true'):
			frame_to[var_to][1] = 'true'
		else:
			frame_to[var_to][1] = 'false'
		frame_to[var_to][0] = 'bool'
	elif(operation == 'not'):
		if(op1 == 'true'):
			frame_to[var_to][1] = 'false'
		else:
			frame_to[var_to][1] = 'true'
		frame_to[var_to][0] = 'bool'
	elif(operation == 'concat'):
		frame_to[var_to][0] = 'string'
		frame_to[var_to][1] = op1 + op2
	inst_counter+=1

def SUB(arg_list):
	ADD(arg_list, '-')	
	
def MUL(arg_list):
	ADD(arg_list, '*')	
	
def IDIV(arg_list):
	ADD(arg_list, '/')	
	
def LT(arg_list):
	ADD(arg_list, '<')	
	
def GT(arg_list):
	ADD(arg_list, '>')	
	
def EQ(arg_list):
	ADD(arg_list, '=')	
	
def AND(arg_list):	
	ADD(arg_list, 'and')	

def OR(arg_list):	
	ADD(arg_list, 'or')	

def NOT(arg_list):
	ADD(arg_list, 'not')	
	
def	INT2CHAR(arg_list):
	global inst_counter
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	symbol = check_symbol(arg_list[2])
	check_operand_type('int', symbol[0], arg_list[2])
	try:
		frame_to[var_to][1] = chr(int(symbol[1]))
	except:
		print("ERROR: Wrong string operation!", file=sys.stderr)
		sys.exit(WRONG_STRING_OPERATION)
	frame_to[var_to][0] = 'string'
	inst_counter+=1

def STRI2INT(arg_list, getchar = None):
	global inst_counter
	
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	symbol1 = check_symbol(arg_list[2])
	check_operand_type('string', symbol1[0], arg_list[2])
	#string z ktoreho sa ma vybrat znak
	string_from = symbol1[1]
	
	symbol2 = check_symbol(arg_list[3])
	check_operand_type('int', symbol2[0], arg_list[3])
	#pozicia vybraneho znaku
	pos = int(symbol2[1])	
	
	#kontrola pozicie ci nepresahuje dlzku stringu
	length = len(string_from) - 1
	if(length < pos or pos < 0):
		print("ERROR: Wrong string operation!", file=sys.stderr)
		sys.exit(WRONG_STRING_OPERATION)
	if(getchar == None):
		frame_to[var_to][1] = ord(string_from[pos])
		frame_to[var_to][0] = 'int'
	if(getchar == True):
		frame_to[var_to][1] = string_from[pos]
		frame_to[var_to][0] = 'string'	
	inst_counter+=1

def READ(arg_list):
	global inst_counter
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	#precitanie vstupu a pokus o pretypovanie na pozadovany typ
	try:
		line = input()
		if(arg_list[2][1] == 'int'):
			frame_to[var_to][0] = 'int'
			frame_to[var_to][1] = int(line)
		elif(arg_list[2][1] == 'bool'):
			if(line.lower()	== 'true'):
				frame_to[var_to][1] = 'true'
			else:
				frame_to[var_to][1] = 'false'
			frame_to[var_to][0] = 'bool'
		elif(arg_list[2][1] == 'string'):
			frame_to[var_to][0] = 'string'
			frame_to[var_to][1] = line
	#ak sa nepodari pretypovanie je do premennej nastavena implicitna
	# hodnota podla pozadovaneho typu	
	except:
		if(arg_list[2][1] == 'int'):
			frame_to[var_to][0] = 'int'
			frame_to[var_to][1] = 0	
		if(arg_list[2][1] == 'bool'):
			frame_to[var_to][0] = 'bool'
			frame_to[var_to][1] = 'false'
		if(arg_list[2][1] == 'string'):
			frame_to[var_to][0] = 'string'
			frame_to[var_to][1] = ''
	inst_counter+=1

def WRITE(arg_list):
	global inst_counter	
	symbol = check_symbol(arg_list[1])
	if(symbol[0] == 'int'):
		symbol[1] = int(symbol[1])
	#vypis zadanej hodnoty
	print(symbol[1])
	inst_counter+=1

def CONCAT(arg_list):
	ADD(arg_list, 'concat')

def STRLEN(arg_list):
	global inst_counter
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	symbol = check_symbol(arg_list[2])
	check_operand_type('string', symbol[0], arg_list[2])
	frame_to[var_to][0] = 'int'
	frame_to[var_to][1] = len(symbol[1])
	inst_counter+=1

def GETCHAR(arg_list):
	STRI2INT(arg_list, True)	

def SETCHAR(arg_list):
	global inst_counter
	pos = None
	character = None
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	if(frame_to[var_to][0] != 'string'):
		print("ERROR: Wrong operand type!", file=sys.stderr)
		sys.exit(WRONG_OPERAND_TYPE)
	symbol1 = check_symbol(arg_list[2])
	check_operand_type('int', symbol1[0], arg_list[2])
	#pozicia na ktorej sa ma nastavit novy znak
	pos = int(symbol1[1])
	
	#kontrola ci je pozicia v ramci daneho stringu validna
	length = len(frame_to[var_to][1]) - 1 
	if(length < pos or pos < 0):
		print("ERROR: Wrong string operation!", file=sys.stderr)
		sys.exit(WRONG_STRING_OPERATION)
	
	symbol2 = check_symbol(arg_list[3])
	check_operand_type('string', symbol2[0], arg_list[3])
	if(symbol2[1] == ''):
		sys.exit(WRONG_STRING_OPERATION)
	#znak ktory sa ma nastavit
	character = symbol2[1][0]
	#nastavenie znaku 'character' na poziciu 'pos'
	frame_to[var_to][1]	= frame_to[var_to][1][:pos] + character + frame_to[var_to][1][(pos+1):]
	inst_counter+=1

def TYPE(arg_list):
	global inst_counter	
	frame_to = check_frame(arg_list[1][1].split("@")[0])
	var_to = check_variable(arg_list[1][1])
	#nastavenie typu zadaneho symbolu do zadanej premennej 
	if(arg_list[2][0] == 'var'):
		frame_from = check_frame(arg_list[2][1].split("@")[0])
		var_from = check_variable(arg_list[2][1])
		if(frame_from[var_from][0] == None):
			frame_to[var_to][1] = ''
		else:
			frame_to[var_to][1] = frame_from[var_from][0]	
	else:
		frame_to[var_to][1] = arg_list[2][0]
	frame_to[var_to][0] = 'string'
	inst_counter+=1

def LABEL(arg_list):
	global inst_counter	
	inst_counter+=1

def JUMP(arg_list):
	global inst_counter
	#skok na pozdaovany label ak neexistuje ide o chybu
	if(labels.get(arg_list[1][1]) == None):
		print("ERROR: Label does not exists!", file=sys.stderr)
		sys.exit(SEMANTIC_ERR)
	else:
		inst_counter = labels.get(arg_list[1][1])

def JUMPIFEQ(arg_list, neq = None):
	global inst_counter
	op2 = None
	op2_type = None	
	if(labels.get(arg_list[1][1]) == None):
		print("ERROR: Label does not exists!", file=sys.stderr)
		sys.exit(SEMANTIC_ERR)
	symbol1 = check_symbol(arg_list[2])
	op1 = symbol1[1]
	op1_type = symbol1[0]

	symbol2 = check_symbol(arg_list[3])
	op2 = symbol2[1]
	op2_type = symbol2[0]	

	check_operand_type(op1_type, op2_type, arg_list[3])
	#JUMPIFEQ
	if(op1 == op2 and neq == None):
		inst_counter = labels.get(arg_list[1][1])
	#JUMPIFNEQ
	elif(op1 != op2 and neq == True):
		inst_counter = labels.get(arg_list[1][1])
	#NO_JUMP
	else:
		inst_counter+=1

def JUMPIFNEQ(arg_list):
	JUMPIFEQ(arg_list, True)		
	
def DPRINT(arg_list):
	global inst_counter
	print(arg_list, file=sys.stderr)	
	inst_counter+=1

def BREAK(arg_list):
	global inst_counter
	print('_____________________________________________',file = sys.stderr)
	print('GF:', GF, file=sys.stderr)
	print('TF:', TF, file=sys.stderr)
	print('LF:', LF, file=sys.stderr)
	print('frame_stack:', frame_stack, file=sys.stderr)
	print('call_stack:', call_stack, file=sys.stderr)
	print('data_stack:', data_stack, file=sys.stderr)
	print('IP:', inst_counter, file=sys.stderr)
	print('_____________________________________________',file = sys.stderr)
	inst_counter+=1

# funkcia vrati pocet inicializovanych premennych vo vsetkych ramcoch
def init_variable_count():
	cnt = 0
	if(LF != None):
		for key in LF:
			if(LF[key][0] != None):
				cnt+=1
	if(TF != None):
		for key in TF:
			if(TF[key][0] != None):
				cnt+=1
	for key in GF:
		if(GF[key][0] != None):
			cnt+=1
	return cnt

# funkcia vykonvajuca instukcie a zaroven zhromazduje 
# statistiky o vykonavanom kode, ktore nasledne vypisuje
# do suboru
def execute_instructions():
	inst_count = len(root)
	counter_insts = 0
	counter_vars = 0
	while(inst_counter <= inst_count):
		xml_inst[inst_counter][0](xml_inst[inst_counter][1])
		counter_insts+=1
		counter_vars = max(counter_vars, init_variable_count())
	#vypis statistik do suboru (rozsirenie)
	if(vars_e or insts_e):
		try:
			f = open(str(stats_file), 'w')
		except:
			sys.exit(FILE_ERROR_OUT)
		if(vars_e != 0 and insts_e == 0):
			f.write(str(counter_vars)+'\n')
		if(insts_e != 0 and vars_e == 0):
			f.write(str(counter_insts)+'\n')
		if(insts_e != 0 and vars_e != 0):
			if(insts_e < vars_e):	
				f.write(str(counter_insts)+'\n')
				f.write(str(counter_vars)+'\n')
			else:
				f.write(str(counter_vars)+'\n')
				f.write(str(counter_insts)+'\n')

# funkcia skontroluje argumenty prikazoveho riadku
def check_arguments():
	global insts_e
	global vars_e
	global stats_file
	help_f = False
	#zistenie pozicie argumentov na prikazovom riadku
	i = 0
	for op in sys.argv:
		if op in ['-v', '--vars']:
			vars_e = i		
		elif op in ['-i', '--insts']:
			insts_e = i
		elif op in ['-h', '--help']:
			help_f = True
			if(len(sys.argv) > 2):
				sys.exit(ARGUMENT_ERROR)
		i+=1
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--source', required = True, help='input XML file containing source code', metavar = ('SOURCE_FILE'))
	parser.add_argument('-t', '--stats', help='output stats file', metavar = ('FILE'))
	parser.add_argument('-i', '--insts', action = 'store_true', help='number of executes instructions')
	parser.add_argument('-v', '--vars', action = 'store_true', help='max number of vars in frames')
	try:
		args = parser.parse_args()
	except:
		if(help_f):
			sys.exit(0)
		else:
			print("ERROR: Argument error!", file=sys.stderr)
			sys.exit(ARGUMENT_ERROR)
	#kontrola ci je kombinacia zadanych argumentov validna
	if((args.insts == True or args.vars == True) and args.stats == None):
		print("ERROR: Argument error!", file=sys.stderr)
		sys.exit(ARGUMENT_ERROR)
	if(args.stats != None):
		stats_file = args.stats

	return args

# funkcia kontroluje korenovy element zdrojoveho xml suboru
def check_xml_head(root):
	attr_count = len(root.attrib)
	if(root.tag != 'program'):
		return XML_ERR 

	language = root.get('language')
	name = root.get('name')
	desc = root.get('description')

	if(attr_count > 3):
		return XML_ERR
	if(attr_count == 1):
		if(language == None):
			return XML_ERR
	if(attr_count == 2):
		if(name == None and desc == None):
			return XML_ERR
		elif(language == None):
			return XML_ERR
	if(attr_count == 3):
		if(language == None or name == None or desc == None):
			return XML_ERR 
	if(language != 'IPPcode18'):
		return SYNLEX_ERR
	return 0
	
#####################################################################################
#	MAIN
#####################################################################################

#kontrola vstupneho suboru
arguments = check_arguments()
source = arguments.source
if(not(os.path.isfile(source))):
	print('ERROR: File',source,'is not valid!',file=sys.stderr)
	sys.exit(FILE_ERROR_IN)

#ziskanie instrukcii z xml suboru
try:
	tree = ET.parse(source)
except:
	print('ERROR: Wrong xml file!', file=sys.stderr)
	sys.exit(XML_ERR)
root = tree.getroot()

#kontrola xml hlavicky
ret = check_xml_head(root)
if(ret != 0):
	sys.exit(ret)

for inst in root: 
	#kontorla tagu instruction
	if(len(inst.attrib) != 2 or inst.tag != 'instruction'):
		print("ERROR: XML error!", file=sys.stderr)
		sys.exit(XML_ERR)
	inst_name = inst.get('opcode')
	inst_order = inst.get('order')
	#kontrola ci boli zadane vsetky atributy
	if(inst_name == None or inst_order == None):
		print("ERROR: XML error!", file=sys.stderr)
		sys.exit(XML_ERR)
	#lexikalna kontrola atributu 'order'
	try:
		int(inst_order)
	except:
		print("ERROR: SYNLEX error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)
	#kontrola poctu argumentov v xml subore 
	if(len(inst) > 3):
		print("ERROR: XML error!", file=sys.stderr)
		sys.exit(XML_ERR)
	#kontrola spravnosti operacneho kodu instrukcie
	if(instructions.get(inst_name) == None):
		print("ERROR: SYNLEX error!", file=sys.stderr)
		sys.exit(SYNLEX_ERR)
	#kontrola ci sedi pocet argumentov pre zadanu instrukciu
	if(len(instructions[inst_name]) != len(inst)):
		print("ERROR: XML error!", file=sys.stderr)
		sys.exit(XML_ERR)	
	#pridanie labelu do slovnika labelov
	if(inst_name == 'LABEL'):
		if(labels.get(inst[0].text) != None):
			print("ERROR: Semantic error Label redefinition!", file=sys.stderr)
			sys.exit(SEMANTIC_ERR)
		labels[inst[0].text] = int(inst_order)
	arg_list_m = {}
	arg_num = 0
	for x in range(len(inst)):
		#kontrola tagu 'type'
		if(re.match(r'^arg[0-9]$',inst[x].tag) == None):
			print("ERROR: XML error!", file=sys.stderr)
			sys.exit(XML_ERR)
		#ziskanie cisla argumentu
		if(arg_num != int(inst[x].tag[-1])):
			arg_num = int(inst[x].tag[-1])
		else:
			print("ERROR: XML error!", file=sys.stderr)
			sys.exit(XML_ERR)
		#ak je cislo argumentu vacsie ako pocet argumentov instrukcie
		if(arg_num > len(instructions[inst_name])):
			print("ERROR: XML error!", file=sys.stderr)
			sys.exit(XML_ERR)
		#ak tag type nema prave jeden atribut => chyba
		if(len(inst[x].attrib) != 1):
			print("ERROR: XML error!", file=sys.stderr)
			sys.exit(XML_ERR)
		arg_t = inst[x].get('type')
		if(arg_t == None):
			print("ERROR: XML error!", file=sys.stderr)
			sys.exit(XML_ERR)
		#ak sa jedna o nepoveleny typ => chyba
		if(arg_t != 'var' and arg_t != 'int' and arg_t != 'bool' and arg_t != 'string' and arg_t != 'label' and arg_t != 'type'):
			print("ERROR: Syntax/lex error!", file=sys.stderr)
			sys.exit(SYNLEX_ERR)
		#lexikalna kontrola argumentov
		instructions[inst_name][int(arg_num)](inst[x].text)

		if(inst[x].text == None):
			inst[x].text = ''
		if(arg_t == 'string'):
			inst[x].text = remove_escape(inst[x].text)
		#slovnik s argumentami
		arg_list_m[int(arg_num)] = {0:arg_t, 1:inst[x].text}
	#funkcia odpovedajuca danej instrukcii
	inst_fun = eval(inst.get('opcode'))
	#slovnik so spracovanymi instrukciami
	xml_inst[int(inst_order)] = {0:inst_fun, 1:arg_list_m}
#kontrola ci sa v subore nachadzaju vsetky instrukcie (podla opcode)
x=1
for i in sorted(xml_inst.keys()):
		if(x != i):
			print('instruction',x,'is missing', file=sys.stderr)
			sys.exit(100)
		x+=1
#spustenie interpretacie
execute_instructions()
