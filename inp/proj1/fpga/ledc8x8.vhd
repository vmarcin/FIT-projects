-- INP 1.projekt
-- Riadenie maticoveho displeja
-- Vladimir Marcin 2-BIT xmarci10

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity ledc8x8 is
port ( 
	RESET		: in std_logic;
	SMCLK		: in std_logic;
	ROW		: out std_logic_vector (0 to 7);
	LED		: out std_logic_vector (0 to 7)
		 );
end ledc8x8;

architecture main of ledc8x8 is
	
	signal ce	: std_logic; -- clock enable pre zmenu aktivneho riadku
	signal ce_cnt	: std_logic_vector(7 downto 0); -- citac na zmenu frekvence pre aktivaciu riadkov
	signal rows	: std_logic_vector(7 downto 0); -- vektor riadkov
begin
	-- zmena frekvencie pre aktivaciu riadkov 	
	clk_control: process(SMCLK, RESET)
	begin
		-- asynchronny reset
		if (RESET='1') then 
			ce_cnt <= (others => '0');
		-- nastupna hrana citac sa zvysi o jedna
		elsif (SMCLK'event) and (SMCLK='1') then
			ce_cnt <= ce_cnt + 1;
		end if;
	end process clk_control;
	-- signal ce (clock enable) sa nastavi na jedna ak citac napocital do 255
	ce <= '1' when ce_cnt = "11111111" else '0';

	-- rotacia riadkov --
	rotate_rows: process(RESET, SMCLK, ce)
	begin
		-- asynchronny reset
		if (RESET = '1') then
			rows <= "10000000";
		-- zmena aktivacie riadku (posunieme jednotku v ramci vektoru rows)
		elsif (SMCLK'event) and (SMCLK = '1') and (ce = '1') then
			rows <= rows(0) & rows(7 downto 1);
		end if;
	end process rotate_rows;
	
	ROW <= rows;
		
	-- dekodovanie riadkov --
	decode_rows: process(rows)
	begin
		-- rosvietenie led podla aktualneho riadka
		case rows is
			when "10000000" => LED <= "01110111";
			when "01000000" => LED <= "01110111";
			when "00100000" => LED <= "10101111";
			when "00010000" => LED <= "11011111";
			when "00001000" => LED <= "11101110";
			when "00000100" => LED <= "11100100";
			when "00000010" => LED <= "11101010";
			when "00000001" => LED <= "11101110";
			when others	=> LED <= "11111111";
		end case;
	end process decode_rows;
end main;
