<?php
define( "ARGUMENT_ERROR", 10 );
define( "SUCCESS", 0 );
define( "FILE_ERROR", 12 );

//globalne premenne
$directory = getcwd();
$parse_script = 'parse.php';
$int_script = 'interpret.py';
$recursive_flag = 0;
$test_ok = 0;
$dir_test_ok = 0;

//xml inicializacia
$html = xmlwriter_open_memory();
xmlwriter_set_indent( $html, 1 );
xmlwriter_set_indent_string( $html, '  ' );
$stats = xmlwriter_open_memory();
xmlwriter_set_indent( $stats, 1 );
xmlwriter_set_indent_string( $stats, '  ' );


/**
 * Funkcia vlozi data do riadku tabulky
 *
 * @param      <type>	$resource  	xml do, ktoreho sa maju vlozit data
 * @param      <type>	$style   	styl textu, ktory obsahuje bunka tabulky
 * @param      <type>	$text    	text, ktory obsahuje bunka tabulky
 */
function create_td(&$resource , $style, $text) {
	xmlwriter_start_element($resource, 'td');
	xmlwriter_start_attribute($resource, 'style');
	xmlwriter_text($resource, $style);
	xmlwriter_end_attribute($resource);
	xmlwriter_text($resource, $text);
	xmlwriter_end_element($resource);
}

/**
 * Funkcia vytlaci html dokument na vystyp
 */
function html_to_output(){
	global $html, $stats;
	echo "<!DOCTYPE html>\n";
	echo xmlwriter_output_memory( $html );
	echo xmlwriter_output_memory( $stats );
	echo "</body>\n";
}


/**
 * Funkcia rekurzivne prehlada adresar 'dir' a vrati pole vsetkych najdenych .src suborov
 *
 * @param      <type>	$dir    adresar, ktory sa bude prehladavat
 *
 * @return     array   	pole obsahujuce vsetky najdene .src subory
 */
function recursive_scan( $dir ) {
	$files = array();
	$it = new RecursiveDirectoryIterator( $dir );
	foreach(new RecursiveIteratorIterator( $it ) as $key => $value) {
		if( preg_match('/.+(.src)$/', $key) ) 
			$array = array_push( $files, $key );
	}
	return $files;
}


/**
 * Funkcia vytvori subor s nazvom rovnakym ako ma subor file ale jeho pripony nahradi priponou
 * 'suffix'
 *
 * @param      string  $suffix  pripona noveho suboru
 * @param      <type>  $file    subor s priponou .src, jeho nazov bude pouzity pre novo vytvoreny subor
 *
 * @return     <type>  vrati nazov novo vytvoreneho suboru
 */
function create_files( $suffix, $file ) {
	$tmp = preg_replace( '/(src)$/', $suffix, $file );
	if( !glob( $tmp ) ) {
		if( $suffix == 'rc' ) {
			if( !( $fp = fopen( $tmp,'w' ) ) ) {
				exit( FILE_ERROR );
			}else{
				if( !fwrite( $fp, '0') ) exit( FILE_ERROR );
				fclose( $fp );
			}
		}else {
			touch( $tmp );
		}
	}
	return $tmp;
}


/**
 * Funkcia, ktora spocita statistiku pre dany adresar (percentualnu uspesnost testov)
 *
 * @param      <type>   $dir    adresar pre, ktory sa vytvara statistika
 * @param      integer  $count  pocet testov v adresari
 */
function directory_stats($dir, $count) {
	global $stats, $dir_test_ok;
	xmlwriter_start_element( $stats, 'tr' );
	
	create_td($stats, 'padding: 0 50px 0 0;', str_replace(getcwd(), '.', $dir));
	$percent = ($dir_test_ok / $count)*100;
	$percent = round($percent, 2);
	if( $percent == 100 )
		create_td($stats, 'padding:0 10px 0 0;color:green', '100%');
	else
		create_td($stats, 'padding:0 10px 0 0;color:red', "$percent%");

	xmlwriter_start_element( $stats, 'td' );
	xmlwriter_text( $stats,  "($dir_test_ok/$count)" );
	xmlwriter_end_element( $stats );

	xmlwriter_end_element( $stats );
}

/*
 * osetrenie argumentov (podpora pre long/short options)
 */
$longopts = array ( "help", "directory:", "recursive", "parse-script:", "int-script:");
$options = getopt ( 'hd:rp:i:', $longopts );
foreach( $argv as $key=>$opt ) {
	if(!$key) continue;
	$option_name = preg_replace( '/^-+|=.*/', '', $opt ); 
	if( !array_key_exists( $option_name, $options ) ) {
		fwrite( STDERR, "ERROR: Unrecognized argument '$opt'!\n");
		exit( ARGUMENT_ERROR );
	}
	if( count( $options[$option_name] ) > 1 || ( array_key_exists( $option_name[0], $options ) && $option_name != $option_name[0] )) {
		fwrite( STDERR, "ERROR: Wrong argument '$opt'!\n");
		exit( ARGUMENT_ERROR );
	}
	switch( $option_name[0] ) {
		case 'h':
			if( $argc != 2) {
				fwrite ( STDERR, "ERROR: You musn't combine anything with '--help/-h'!\n" );
				exit( ARGUMENT_ERROR);
			}else {
				echo "NAME\n\ttest.php\n";
				echo "DESCRIPTION\n\tScript serves for automatic testing parse.php and interpret.py.\n";
				echo "OPTIONS\n";
				echo "\t--help print help\n";
				echo "\t--directory=path directory which contain tests (pwd by default)\n"; 
				echo "\t--recursive tests will be recursively in all subdirectories\n";
				echo "\t--parse-script=file file wich contain php script for analysis\n\t  IPPcode18 source code (parse.php by default)\n";
				echo "\t--int-script=file file which contain python script for interpret\n\t  XML representation of IPPcode18 (interpret.py by default)\n";
				exit( SUCCESS );
			}
		case 'd':
			$directory = rtrim($options[$option_name],'/');
			if( !is_dir($directory) ) {
				fwrite( STDERR, "ERROR: Directory $directory doesn't exist!\n" );
				exit( FILE_ERROR );
			}
			break;
		case 'r':
			$recursive_flag = 1;
			break;
		case 'p':
			$parse_script = $options[$option_name];
			if( !is_file($parse_script) ) {
				fwrite( STDERR, "ERROR: $parse_script is not a file\n");
				exit( FILE_ERROR );
			}else if( !preg_match('/.+(.php)$/', $parse_script) ) {
				fwrite( STDERR, "ERROR: $parse_script is not a php file\n");
				exit( FILE_ERROR );
			}
			break;
		case 'i':
			$int_script = $options[$option_name];
			if( !is_file($int_script) ) {
				fwrite( STDERR, "ERROR: $int_script is not a file\n");
				exit( FILE_ERROR );
			}else if( !preg_match('/.+(.py)$/', $int_script) ) {
				fwrite( STDERR, "ERROR: $parse_script is not a python file\n");
				exit( FILE_ERROR );
			}		
			break;
	}
}

/*
 * zaciatok html dokumentu
 */
xmlwriter_start_element( $html, 'head' );
xmlwriter_start_element( $html, 'title' );
xmlwriter_text( $html, 'Test Report' );
xmlwriter_end_element( $html );
xmlwriter_end_element( $html );
xmlwriter_start_element( $html, 'body' );
xmlwriter_start_attribute( $html, 'style' );
xmlwriter_text( $html, 'font-family:Courier New' );
xmlwriter_end_attribute( $html );
xmlwriter_start_element( $html, 'h1' );
xmlwriter_text( $html, 'Test Report' );
xmlwriter_end_element( $html );
xmlwriter_start_element( $html, 'p' );
xmlwriter_start_element( $html, 'b' );
xmlwriter_text( $html, 'pwd:' );
xmlwriter_end_element( $html );
xmlwriter_text( $html, getcwd() );
xmlwriter_end_element( $html );
xmlwriter_start_element( $html, 'hr' );
xmlwriter_end_element( $html );

/*
 * najdenie testov (suborov s priponov .src) 
 * ak bol zadany argument -r dany adresar budeme prehladavat rekurzivne
 * inak sa testy hladaju v aktualnom adresari
 */
if( $recursive_flag ){
	$files = recursive_scan( $directory );
}
else{
	$files = glob( $directory.'/*.src' );
}

/*
 * ziadne testy sa nenasli do HTML napisme informaciu o tom ze sme nenasli testy
 * a ukoncime skript
 */
if(!count($files)) {
	xmlwriter_start_element($html, 'p');
	xmlwriter_start_attribute($html, 'style');
	xmlwriter_text($html, 'color:red;');
	xmlwriter_end_attribute($html);
	xmlwriter_text($html, 'No tests found!');
	xmlwriter_end_element($html);
	html_to_output($html);
	exit(SUCCESS);
}
xmlwriter_start_element( $html, 'table' );

/*
 * na organizaciu testov v jednotlivych aresaroch je pouzite dvojrozmerne pole
 * kde adresar prestavuje kluc do asociativneho pole a hodnota na ktoru tento kluc odkazuje je pole 
 * v ktorom sa nachadzaju testovane subory v danom adresari
 */
foreach($files as $key=>$file) {
	$md_files[dirname($file)][] = $file;
}

/*
 * vytvorenie docasnych suborov pre vystup z jednotlivych skriptov
 */
$parser_output = tmpfile();
$meta_data = stream_get_meta_data( $parser_output );
$tmp_parse = $meta_data["uri"];

$int_output = tmpfile();
$meta_data = stream_get_meta_data( $int_output );
$tmp_int = $meta_data["uri"];
	
xmlwriter_start_element( $stats, 'table');
xmlwriter_start_attribute( $stats, 'style');
xmlwriter_text( $stats, 'padding:0 0 0 50px;' );
xmlwriter_end_attribute( $stats );


/*
 * testovanie samotnych suborov
 */
foreach($md_files as $key=>$dir) {
	xmlwriter_start_element($html, 'tr');
	create_td($html, 'color:blue', str_replace(getcwd(),'.', $key));
	xmlwriter_end_element($html);
	
	natsort($dir);
	foreach($dir as $file) {
		xmlwriter_start_element($html, 'tr');
		create_td($html, 'padding:0 100px 0 50px', basename($file, ".src"));
	
		$out_file = create_files( 'out', $file );
		$in_file = create_files( 'in', $file );
		$rc_file = create_files( 'rc', $file );
		
		if ( !( $fp_rc = fopen( $rc_file, 'r') ) ) {
			exit( FILE_ERROR);
		}
		
		//ziskanie ocakavaneho navratoveho kodu
		$rc = trim( fgets( $fp_rc ) );
		
		$parse_cmd='php5.6 '.$parse_script.' <'.$file.' >'.$tmp_parse.'; echo $?;';

		//ziskanie navratoveho kodu skriptu '$parse_script'  
		$parse_status = rtrim( shell_exec( $parse_cmd ) );

		if( $parse_status != '0') {
			if( $rc == $parse_status ) {
				$test_ok+=1;
				$dir_test_ok+=1;
				create_td($html, 'padding:0 50px 0 0; color:green', 'OK');
			}else {
				create_td($html, 'padding:0 50px 0 0; color:red', 'FAILURE');
				create_td($html, 'padding:0 50px 0 0;', "exit code expected=$rc, returned=$parse_status");
			}
			fclose( $fp_rc );
			xmlwriter_end_element($html);
			continue;
		}

		$int_cmd='python3.6 '.$int_script.' --source='.$tmp_parse.'<'.$in_file.' >'.$tmp_int.'; echo $?;';
		
		//ziskanie navratoveho kodu skriptu '$int_script'
		$int_status = rtrim( shell_exec( $int_cmd ) );

		if( $rc == $int_status ) {
			if( $int_status == '0' ) {
				$diff_cmd = "diff $tmp_int $out_file; echo $?;";
				$diff_status = rtrim( shell_exec( $diff_cmd ) );
				if( $diff_status == 0 ) { 
					$test_ok+=1;
					$dir_test_ok+=1;
					create_td($html, 'padding:0 50px 0 0;color:green', 'OK');
				}
				else {
					create_td($html, 'padding:0 50px 0 0;color:red', 'FAILURE');
					create_td($html, 'padding:0 50px 0 0;', "unexpected interpret output look at $out_file");
				}
			}else {
				$test_ok+=1;
				$dir_test_ok+=1;
				create_td($html, 'padding:0 50px 0 0;color:green', "OK");
			}
		}else {
			create_td($html, 'padding:0 50px 0 0;color:red', "FAILURE");
			create_td($html, 'padding:0 50px 0 0;', "exit code expected=$rc, returned=$int_status");
		}
		xmlwriter_end_element($html);	
	}
	directory_stats($key, count($dir));
	$dir_test_ok = 0;	
}

xmlwriter_end_element( $stats );
xmlwriter_end_element($html);
xmlwriter_start_element( $html, 'hr' );
xmlwriter_end_element( $html );
/*
 * celkova statistika testov (percentualna uspesnost)
 */
$all = count($files);
$per = ($test_ok/$all) * 100;
$per = round($per,2);
xmlwriter_start_element($html, 'p');
xmlwriter_start_element($html, 'b');
xmlwriter_text($html, "RESULTS: $per% ($test_ok/$all)");
xmlwriter_end_element($html);
xmlwriter_end_element($html);

html_to_output();
?>
