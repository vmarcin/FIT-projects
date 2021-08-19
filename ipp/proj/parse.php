<?php
mb_internal_encoding( "UTF-8" );

define( "FILE_ERROR", 12);
define( "ARGUMENT_ERROR", 10 );
define( "INTERNAL_ERROR", 99 );
define( "SYNTAX_ERROR", 21 );
define( "LEXICAL_ERROR", 21 );
define( "SUCCESS", 0 );

/*
 * globalne premenne
 */ 
$comments = 0;
$comments_flag = 0;
$loc = 0;
$loc_flag = 0;
$line_num = 0;
$stats_file = '';

/*
 * xml inicializacia
 */
$output = xmlwriter_open_memory();
xmlwriter_set_indent( $output, 1 );
xmlwriter_set_indent_string( $output, '  ' );

/*
 * asociativne pole ktore obsahuje informacie o instrukciach
 * (pocet a typ argumentov instrukcie)
 * tieto informacie su ulozene vo forme nazvov funkcii pomocou 
 * ktorych sa neskor kontroluje spravnost argumentov
 */
$instructions = array(
	"MOVE"=> array( 1 => "check_var", "check_symb" ),
	"CREATEFRAME"=> array(),
	"PUSHFRAME"=> array(),
	"POPFRAME"=> array(),
	"DEFVAR"=> array( 1 => "check_var" ),
	"CALL"=> array( 1 => "check_label" ),
	"RETURN"=> array(),
	"PUSHS"=> array( 1 => "check_symb" ),
	"POPS"=> array( 1 => "check_var" ),
	"ADD"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"SUB"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"MUL"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"IDIV"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"LT"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"GT"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"EQ"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"AND"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"OR"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"NOT"=> array( 1 => "check_var", "check_symb" ),
	"INT2CHAR"=> array( 1 => "check_var", "check_symb" ),
	"STRI2INT"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"READ"=> array( 1 => "check_var", "check_type" ),
	"WRITE"=> array( 1 => "check_symb" ),
	"CONCAT"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"STRLEN"=> array( 1 => "check_var", "check_symb" ),
	"GETCHAR"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"SETCHAR"=> array( 1 => "check_var", "check_symb", "check_symb" ),
	"TYPE"=> array( 1 => "check_var",  "check_symb" ),
	"LABEL"=> array( 1 => "check_label" ),
	"JUMP"=> array( 1 => "check_label" ),
	"JUMPIFEQ"=> array( 1 => "check_label", "check_symb", "check_symb" ),
	"JUMPIFNEQ"=> array( 1 => "check_label", "check_symb", "check_symb" ),
	"DPRINT"=> array( 1 => "check_symb" ),
	"BREAK"=> array(),
);

/**
 * Funkcia skontroluje lexikalnu spravnost typu
 *
 * @param      string  $type   kontrolovany typ
 *
 * @return     integer  1-OK 0-FAIL
 */
function check_type( $type ) {
	global $output;
	xmlwriter_text( $output, 'type' );
	xmlwriter_end_attribute( $output );
	xmlwriter_text( $output, $type );
	return preg_match( '/^(int|bool|string)$/', $type );
}

/**
 * Funkcia skontroluje lexikalnu spravnost navesti
 *
 * @param      string  $label  kontrolovane navesti
 *
 * @return     integer  1-OK 0-FAIL
 */
function check_label( $label ) {
	global $output;
	xmlwriter_text( $output, 'label' );
	xmlwriter_end_attribute( $output );
	xmlwriter_text( $output, $label );
	return preg_match( '/^[a-zA-Z-_$&%*][[:alnum:]-_$&%*]*$/', $label );
}

/**
 * Funkcia skontroluje lexikalnu spravnost premennej
 *
 * @param      string  $var    kontrolovana premenna
 *
 * @return     integer  1-OK 0-FAIL
 */
function check_var( $var) {
	global $output;
	xmlwriter_text( $output, 'var' );
	xmlwriter_end_attribute( $output );
	xmlwriter_text( $output, $var );
	return preg_match( '/^[LTG]F@[a-zA-Z-_$&%*][[:alnum:]-_$&%*]*$/', $var );
}


/**
 * Funkcia skontroluje lexikalnu spravnost konstanty
 *
 * @param      string  $constant  kontrolovana konstanta
 *
 * @return     integer  1-OK 0-FAIL
 */
function check_constant( $constant ) {
	$const_int = preg_match( '/^int@(-|\+)?\d+$/', $constant );
	$const_bool	= preg_match( '/^bool@(true|false)$/', $constant );
	$const_string	= preg_match( '/^string@([^\s\x5C]|\x5C[0-9]{3})*$/', $constant );
	return ( $const_int or $const_bool or $const_string );
}

/**
 * Funkcia skontroluje lexikalnu spravnost symbolu (symbol = konstanta|premenna)
 *
 * @param      string   $symb   kontrolovany symbol
 *
 * @return     integer  1-OK 0-FAIL
 */
function check_symb( $symb ) {
	global $output;
	if( check_constant( $symb ) ) {
		$arg = preg_split( '/@/', $symb );
		xmlwriter_text( $output, $arg[0] );
		xmlwriter_end_attribute( $output );
		xmlwriter_text( $output, $arg[1] );
		return 1;		
	}else if( check_var( $symb ) ) {
		return 1;
	}
	return 0;
}

/**
 * Funkcia odstrani komentar z riadku '$comm_line' a vrati upraveny riadok
 *
 * @param      string  $comm_line  riadok obsahujuci komentar
 *
 * @return     string  upraveny riadok bez komentara
 */
function strip_comment ( $comm_line ) {
	global $comments;
	$comments +=1;
	$comm_line = preg_replace( '/#.*/', '', $comm_line );
	$comm_line = trim( $comm_line );
	return $comm_line;
}

/*
 * kontrola argumentov (podpora short/long options)
 */
$longopts = array( "help", "stats:", "loc", "comments" );
$options = getopt('hs:lc', $longopts );
foreach( $argv as $key=>$opt ) {
	if( !$key ) continue;
	$option_name = preg_replace( '/^-+|=.*/', '', $opt );
	if( !array_key_exists( $option_name, $options ) ) {
		fwrite( STDERR, "ERROR: Unrecognized argument '$opt'!\n" );
		exit ( ARGUMENT_ERROR );
	}
	if( count( $options[$option_name] ) > 1 || ( array_key_exists( $option_name[0], $options ) && $option_name != $option_name[0] )) {
		fwrite ( STDERR, "ERROR: Wrong argument '$opt'!\n");
		exit ( ARGUMENT_ERROR );
	}
	switch( $option_name[0] ){
		case 'h':
			if( $argc != 2) {
				fwrite( STDERR, "ERROR: You musn't combine anything with '--help/-h'!\n");
				exit( ARGUMENT_ERROR );
			}else{
				echo "NAME\n\tparse.php\n";
				echo "DESCRIPTION\n\tScript checks lexical and syntax correctness of source code in IPPcode18 language and prints the XML program representation to the stdout.\n";
				echo "OPTIONS\n";
				echo "\t--help prints help\n";
				echo "\t--stats=file collection of statictics for the processed source code\n"; 
				echo "\t--loc prints number of line of codes into 'file'\n";
				echo "\t--comments prints number of comments into 'file'\n";
				exit( SUCCESS );
				echo "HELP\n";
				exit( SUCCESS );
			}
		case 'l':
			$loc_flag = $key; 
			break;
		case 'c':
			$comments_flag = $key;
			break;
		case 's':
			$stats_file = $options[$option_name];
			break;
		default:
			exit ( ARGUMENT_ERROR );
	}
}
if ( (($loc_flag || $comments_flag) && $stats_file == '') ) {
	fwrite( STDERR, "ERROR: Wrong arguments combination!\n" );
	exit( ARGUMENT_ERROR );
}

/*
 * odstranenie prazdnych riadkov a komentarov nachadzajucich sa pred hlavickou
 */
do{
	$line = trim( fgets( STDIN ) );
	$line_num+=1;
	if( preg_match( '/#/', $line ) ) {
		$line = strip_comment( $line );
	}
}while( $line == '' && !feof( STDIN ) );

/*
 * lexikalna kontrola hlavicky
 */
if( !preg_match( "/^\.IPPcode18$/i", $line ) ) {
	fwrite( STDERR, "ERROR: Invalid header (line: $line_num)\n" );
	exit(SYNTAX_ERROR);
}


xmlwriter_start_document( $output, '1.0', 'UTF-8' );
xmlwriter_start_element( $output, 'program' );
xmlwriter_start_attribute( $output, 'language' );
xmlwriter_text( $output, 'IPPcode18' );
xmlwriter_end_attribute( $output );

/*
 * kontrola samotneho kodu (cyklus pre kazdu instrukciu)
 */
while( !feof( STDIN ) ) {
	$line = trim( fgets( STDIN ) );
	$line_num+=1;
	if ( $line == '') {
		continue;
	}
	/*
	 * ak riadok obsahuje komentar odstranime ho
	 */
	if( preg_match( '/#/', $line ) ) {
		$line = strip_comment( $line );
		if( $line == '') {
			continue;
		}
	}
	/*
	 * rozdelenie instrukciu na meno a operandy
	 */
	$inst = preg_split( "/[\s]+/", $line );
	$loc += 1;
	$inst_name = strtoupper( $inst[0] );
	/*
	 * kontrola ci ide o validnu instrukciu
	 */
	if( !array_key_exists( $inst_name , $instructions ) ) {
		fwrite( STDERR, "ERROR: Instruction deosn't exist! (line: $line_num)\n" );
		exit( SYNTAX_ERROR );
	}
	/*
	 * kontrola ci sedi pocet operandov instrukcie
	 */
	if( count( $instructions[$inst_name] ) != ( count($inst) - 1 ) ) {
		fwrite( STDERR, "ERROR: Too many/few arguments! (line: $line_num)\n" );
		exit( SYNTAX_ERROR);
	}

	xmlwriter_start_element( $output, 'instruction' );
	xmlwriter_start_attribute( $output, 'order' );
	xmlwriter_text( $output, $loc );
	xmlwriter_end_attribute( $output );
	xmlwriter_start_attribute( $output, 'opcode' );
	xmlwriter_text( $output, $inst_name );
	xmlwriter_end_attribute( $output );

	/*
	 * kontrola lexikalnej spravnosti operandov
	 */
	for( $i=1; $i < count( $inst ); $i++) {
		xmlwriter_start_element( $output, 'arg'.$i );
		xmlwriter_start_attribute( $output, 'type' );
		/*
		 * pomocou premennej '$inst_name' zaindexujeme pole s instrukciami a pomocou premennej '$i'
		 * sa dostaneme k i-temu operandu danej instrukcie a tym sa dostaneme k menu funkciu, ktorej ako 
		 * parameter posleme i-ty operand kontrolovanej instrukcie
		 */
		if( !( $instructions[$inst_name][$i]( $inst[$i] ) ) ) {
			fwrite( STDERR, "ERROR: Operand $i is wrong (line: $line_num)\n" );
			exit( LEXICAL_ERROR );
		}
		xmlwriter_end_element( $output );	
	}
	xmlwriter_end_element( $output );
}
xmlwriter_end_element( $output );

/*
 * ak bol program spusteny s --loc alebo --comments
 * do suboru sa v spravnom poradi zapise pocet instrukcii
 * a pocet komentarov
 */
if( $loc_flag or $comments_flag ) {
	$fp = fopen( $options['stats'], "w" );
	if( $fp == FALSE ) {
		fwrite( STDERR, "ERROR: Can not open file!\n" );
		exit( FILE_ERROR );
	}
	if( $loc_flag and $comments_flag ) {
		if( $loc_flag > $comments_flag ) {
			fwrite( $fp, "$comments\n$loc\n" );
		}else {
			fwrite( $fp, "$loc\n$comments\n" );
		}
	}else if( $loc_flag ) {
		fwrite( $fp, "$loc\n" );
	}else if( $comments_flag) {
		fwrite( $fp, "$comments\n" );
	}
	fclose( $fp );
}
xmlwriter_end_document( $output );
echo xmlwriter_output_memory( $output );
exit( SUCCESS );
?>
