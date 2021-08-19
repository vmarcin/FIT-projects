-- cpu.vhd: Simple 8-bit CPU (BrainLove interpreter)
-- Copyright (C) 2017 Brno University of Technology,
--                    Faculty of Information Technology
-- Author(s): MARCIN Vladimir (xmarci10) 
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

-- ----------------------------------------------------------------------------
--                        Entity declaration
-- ----------------------------------------------------------------------------
entity cpu is
 port (
   CLK   : in std_logic;  -- hodinovy signal
   RESET : in std_logic;  -- asynchronni reset procesoru
   EN    : in std_logic;  -- povoleni cinnosti procesoru
 
   -- synchronni pamet ROM
   CODE_ADDR : out std_logic_vector(11 downto 0); -- adresa do pameti
   CODE_DATA : in std_logic_vector(7 downto 0);   -- CODE_DATA <- rom[CODE_ADDR] pokud CODE_EN='1'
   CODE_EN   : out std_logic;                     -- povoleni cinnosti
   
   -- synchronni pamet RAM
   DATA_ADDR  : out std_logic_vector(9 downto 0); -- adresa do pameti
   DATA_WDATA : out std_logic_vector(7 downto 0); -- mem[DATA_ADDR] <- DATA_WDATA pokud DATA_EN='1'
   DATA_RDATA : in std_logic_vector(7 downto 0);  -- DATA_RDATA <- ram[DATA_ADDR] pokud DATA_EN='1'
   DATA_RDWR  : out std_logic;                    -- cteni z pameti (DATA_RDWR='0') / zapis do pameti (DATA_RDWR='1')
   DATA_EN    : out std_logic;                    -- povoleni cinnosti
   
   -- vstupni port
   IN_DATA   : in std_logic_vector(7 downto 0);   -- IN_DATA obsahuje stisknuty znak klavesnice pokud IN_VLD='1' a IN_REQ='1'
   IN_VLD    : in std_logic;                      -- data platna pokud IN_VLD='1'
   IN_REQ    : out std_logic;                     -- pozadavek na vstup dat z klavesnice
   
   -- vystupni port
   OUT_DATA : out  std_logic_vector(7 downto 0);  -- zapisovana data
   OUT_BUSY : in std_logic;                       -- pokud OUT_BUSY='1', LCD je zaneprazdnen, nelze zapisovat,  OUT_WE musi byt '0'
   OUT_WE   : out std_logic                       -- LCD <- OUT_DATA pokud OUT_WE='1' a OUT_BUSY='0'
 );
end cpu;


-- ----------------------------------------------------------------------------
--                      Architecture declaration
-- ----------------------------------------------------------------------------
architecture behavioral of cpu is

 -- PC 
 signal pc_reg : std_logic_vector(11 downto 0);			-- register PC (program counter) sluzi ako ukazatel do pamati programu
 signal pc_inc : std_logic;					-- ak pc_inc='1' => pc_reg<=pc_reg + 1, PC ukazuje na nasledujucu instrukciu
 signal pc_dec : std_logic;					-- ak pc_dec='1' => pc_reg<=pc_reg - 1, PC ukazuje na predchadzajucu instrukciu 
 -- PTR
 signal ptr_reg : std_logic_vector(9 downto 0);			-- register PTR sluzi ako ukazatel do pamati dat
 signal ptr_inc : std_logic;					-- ak ptr_inc='1' => ptr_reg<=ptr_reg + 1, PTR ukazuje na nasledujucu adresu v pamati
 signal ptr_dec : std_logic;					-- ak ptr_dec='1' => ptr_reg<=ptr_reg - 1, PTR ukazuje na predchadzajucu adresu v pamati
 -- CNT 
 signal cnt_reg 		: std_logic_vector(7 downto 0);	-- register CNT sluzi ku korektnemu urceniu odpovedajuceho zaciatku/konca prikazu while
 signal cnt_inc 		: std_logic;			-- ak cnt_inc='1' => cnt_reg<=cnt_reg + 1
 signal cnt_dec 		: std_logic;			-- ak cnt_dec='1' => cnt_reg<=cnt_reg - 1
 signal cnt_set_one : std_logic;			-- ak cnt_set_one='1' => cnt_reg<=X"01"
 -- dec
 type inst_type is(
   inc_val,     -- +
   dec_val,     -- -
   inc_ptr,     -- >
   dec_ptr,     -- <
   begin_while, -- [
   end_while,   -- ]
   putchar,     -- .
   getchar,     -- ,
   break,       -- ~
   halt,        -- null
   nope         -- comments (hocico okrem prechadzajucich instrukcii)
 );
 signal ireg_dec : inst_type;					-- sluzi k dekodovaniu instrukcie
 -- MX
 signal mx_sel : std_logic_vector(1 downto 0);			-- sluzi k volbe hodnoty zapisovanej do pamate dat
 -- FSM
 type fsm_state is(
 	sidle, sfetch, sdecode, sinc_val0, sinc_val1, 
	sdec_val0, sdec_val1, sinc_ptr, sdec_ptr, 
	sputchar0, sputchar1, sgetchar0, sgetchar1,
	sbreak0, sbreak1, sbreak2, shalt, snope,
	sbegin_while0, sbegin_while1, sbegin_while2, sbegin_while3,
	send_while0, send_while1, send_while2, send_while3, send_while4
 );
 signal pstate : fsm_state;					-- aktualny stav automatu
 signal nstate : fsm_state;					-- nasledujuci stav automatu

begin

 -- PROGRAMOVY CITAC (ukazatel do pamate kodu)
 pc_register : process (RESET, CLK)
 begin
  if (RESET='1') then
    pc_reg <= (others=>'0');
  elsif (CLK'event) and (CLK='1') then
    if (pc_inc='1') then
      pc_reg <= pc_reg + 1;
    elsif (pc_dec='1') then
      pc_reg <= pc_reg - 1;
    end if;
  end if;
 end process pc_register;

 CODE_ADDR <= pc_reg;

 -- UKAZATEL DO PAMATE DAT
 ptr_register : process (RESET, CLK)
 begin
  if (RESET='1') then
    ptr_reg <= (others=>'0');
  elsif (CLK'event) and (CLK='1') then
    if (ptr_inc='1') then
      ptr_reg <= ptr_reg + 1;
    elsif (ptr_dec='1') then
      ptr_reg <= ptr_reg - 1;
    end if;
  end if;
 end process ptr_register;

 DATA_ADDR <= ptr_reg;
 
 -- CNT REGISTER (pre vnorene cykly) 
 cnt_register : process (RESET, CLK)
 begin
  if (RESET='1') then
    cnt_reg <= (others=>'0');
  elsif (CLK'event) and (CLK='1') then
		if (cnt_set_one='1') then
			cnt_reg <= X"01";
		end if;
		if (cnt_inc='1') then
      cnt_reg <= cnt_reg + 1;
    elsif (cnt_dec='1') then
      cnt_reg <= cnt_reg - 1;
    end if;
  end if;
 end process cnt_register;

 -- DEKODER INSTRUKCII
 inst_decoder : process (CODE_DATA)
 begin
  case (CODE_DATA) is
    when X"3E"  => ireg_dec <= inc_ptr;    -- >
    when X"3C"  => ireg_dec <= dec_ptr;    -- <
    when X"2B"  => ireg_dec <= inc_val;    -- +
    when X"2D"  => ireg_dec <= dec_val;    -- -
    when X"5B"  => ireg_dec <= begin_while;-- [
    when X"5D"  => ireg_dec <= end_while;  -- ]
    when X"2E"  => ireg_dec <= putchar;    -- .
    when X"2C"  => ireg_dec <= getchar;    -- ,
    when X"7E"  => ireg_dec <= break;      -- ~
    when X"00"  => ireg_dec <= halt;       -- null
    when others => ireg_dec <= nope;       -- comments
  end case;
 end process inst_decoder;
 
 -- MULTIPLEXOR NA VOLBU HODNOTY ZAPISOVANEJ DO PAMATE DAT
 with mx_sel select
 DATA_WDATA <=  IN_DATA        when "00",
                DATA_RDATA + 1 when "01",
                DATA_RDATA - 1 when "10",
                X"00"          when others;
 
 -- nastavenie aktualneho stavu automatu
 fsm_pstate : process (RESET, CLK)
 begin
  if (RESET='1') then
    pstate <= sidle;
  elsif (CLK'event) and (CLK='1') then
    if (EN='1') then
      pstate <= nstate;
    end if;
  end if;
 end process fsm_pstate;

 --FSM next state logic
 NSL: process(pstate, ireg_dec,IN_VLD, OUT_BUSY, cnt_reg, DATA_RDATA)
 begin
  -- INIT
  CODE_EN <= '0';
  DATA_RDWR <= '0';
  DATA_EN <= '0';
  IN_REQ <= '0';
  OUT_WE <= '0';

  pc_inc <= '0';
  pc_dec <= '0';
  ptr_inc <= '0';
  ptr_dec <= '0';
  cnt_inc <= '0';
  cnt_dec <= '0';
	cnt_set_one <= '0';
  mx_sel <= "11";

  case pstate is
    -- IDLE
    when sidle =>
      nstate <= sfetch;
    -- INSTRUCTION FETCH
    when sfetch =>
      nstate <= sdecode;
      CODE_EN <= '1';
		-- DECODE INSTRUCTION
    when sdecode =>
      case ireg_dec is
        when inc_val => 
          nstate <= sinc_val0;
        when dec_val => 
          nstate <= sdec_val0;
        when inc_ptr => 
          nstate <= sinc_ptr;
        when dec_ptr => 
          nstate <= sdec_ptr;
        when begin_while => 
          nstate <= sbegin_while0;
        when end_while => 
          nstate <= send_while0;
        when putchar => 
          nstate <= sputchar0;
        when getchar =>
          nstate <= sgetchar0;
        when break => 
          nstate <= sbreak0;
        when halt => 
          nstate <= shalt;
        when nope => 
          nstate <= snope;
      end case;
    -- HALT
    when shalt =>
      nstate <= shalt;
    -- NOPE
    when snope =>
      nstate <= sfetch;
      pc_inc <= '1';
    -- INC_PTR
    when sinc_ptr =>
      nstate <= sfetch;
      ptr_inc <= '1';
      pc_inc <= '1';
    -- DEC_PTR
    when sdec_ptr =>
      nstate <= sfetch;
      ptr_dec <= '1';
      pc_inc <= '1';
    -- INC_VAL
    when sinc_val0 => 
      nstate <= sinc_val1;
      DATA_EN <= '1';
      DATA_RDWR <= '0';
    
    when sinc_val1 => 
      nstate <= sfetch;
      DATA_EN <= '1';
      DATA_RDWR <= '1';
      mx_sel <= "01";
      pc_inc <= '1';
    -- DEC_VAL
    when sdec_val0 => 
      nstate <= sdec_val1;
      DATA_EN <= '1';
      DATA_RDWR <= '0';
    
    when sdec_val1 => 
      nstate <= sfetch;
      DATA_EN <= '1';
      DATA_RDWR <= '1';
      mx_sel <= "10";
      pc_inc <= '1';
    -- PUTCHAR
    when sputchar0 => 
     if (OUT_BUSY='1') then
      nstate <= sputchar0;
     else
      DATA_EN <= '1';
      DATA_RDWR <= '0';
      nstate <= sputchar1;
     end if;
    
    when sputchar1 => 
      nstate <= sfetch;
      OUT_WE <= '1';
      OUT_DATA <= DATA_RDATA;
      pc_inc <= '1';
    -- GETCHAR
    when sgetchar0 => 
      IN_REQ <= '1';
      if (IN_VLD/='1') then
        nstate <= sgetchar0;
      else
        nstate <= sgetchar1;
      end if;
    
    when sgetchar1 => 
      nstate <= sfetch;
      DATA_EN <= '1';
      DATA_RDWR <= '1';
      mx_sel <= "00";
      pc_inc <= '1';
      IN_REQ <= '0';
    -- BREAK
    when sbreak0 => 
      nstate <= sbreak1;
      cnt_set_one <= '1';
			pc_inc <= '1';
    
		when sbreak1 =>
			nstate <= sbreak2;
			CODE_EN <= '1';

    when sbreak2 => 
			if (cnt_reg/=X"00") then
      	if (ireg_dec=end_while) then
        	cnt_dec <= '1';
				elsif (ireg_dec=begin_while) then
        	cnt_inc <= '1';
      	end if;
				nstate <= sbreak1;
				pc_inc <= '1';
			else
				nstate <= sfetch;
			end if;
    -- BEGIN_WHILE
    when sbegin_while0 =>
      nstate <= sbegin_while1;
      pc_inc <= '1';
      DATA_EN <= '1';
      DATA_RDWR <= '0';
    
    when sbegin_while1 =>
      if (DATA_RDATA=X"00")  then
        nstate <= sbegin_while2;
        cnt_set_one <= '1';
      else
        nstate <= sfetch;
      end if;
    
    when sbegin_while2 =>
      nstate <= sbegin_while3;
      CODE_EN <= '1';
    
    when sbegin_while3 => 
			if(cnt_reg/=X"00") then
        if (ireg_dec = begin_while) then
          cnt_inc <= '1';
        elsif (ireg_dec = end_while) then
          cnt_dec <= '1';
        end if;
        nstate <= sbegin_while2;
        pc_inc <= '1';	
			else
				nstate <= sfetch;
      end if;
    -- END_WHILE
    when send_while0 =>
      nstate <= send_while1;
      DATA_EN <= '1';
      DATA_RDWR <= '0';
    
    when send_while1 =>
      if (DATA_RDATA=X"00") then
        pc_inc <= '1';
        nstate <= sfetch;
      else  
        nstate <= send_while2;
        cnt_set_one <= '1';
        pc_dec <= '1';
      end if;

    when send_while2 =>
      nstate <= send_while3;
      CODE_EN <= '1';

    when send_while3 =>
      if (cnt_reg/=X"00") then
        if (ireg_dec=end_while) then
          cnt_inc <= '1';
        elsif (ireg_dec=begin_while) then
          cnt_dec <= '1';
        end if;
        nstate <= send_while4;
			else
				nstate <= sfetch;
      end if;

    when send_while4 =>
      nstate <= send_while2;
      if (cnt_reg=X"00") then
        pc_inc <= '1';
      else
        pc_dec <= '1';
      end if;
  end case;
 end process NSL;
end behavioral;
