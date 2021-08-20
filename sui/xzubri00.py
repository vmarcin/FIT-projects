import random
import copy
import logging

from dicewars.client.game.board import Board
from dicewars.client.game.area import Area
from dicewars.ai.utils import possible_attacks, attack_succcess_probability, save_state
from dicewars.ai.utils import probability_of_successful_attack

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, AIDriver
from dicewars.ai.dt.stei import AI as AIstei
from dicewars.ai.dt.wpm_c import AI as AIwpmc

class AI:

    def __init__(self, player_name, board, players_order):
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.players_order = players_order
        self.counter = 1
        self.board = board

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        
        self.board = board
        
        turns = self.possible_turns(nb_moves_this_turn, nb_turns_this_game, time_left)

        if turns:
            turn = turns[0]
            area_name = turn[0]
            value = turn[2][0]
            return BattleCommand(area_name, turn[1])

        return EndTurnCommand()

    def possible_turns(self, nb_moves_this_turn, nb_turns_this_game, time_left):
        turns = []
        """
            Cyklus cez vsetky mozne utoky, ktore je mozne uskutocnit v aktualnom tahu.
            Pre kazdy utok volane expectiminimax.
        """
        for source, target in possible_attacks(self.board, self.player_name):
            atk_prob = probability_of_successful_attack(self.board, source.get_name(), target.get_name())
            """
                Uspesny utok -- aktualizacia boardu (podla pravidiel hry).
            """
            success_board = copy.deepcopy(self.board)
            target_area = success_board.areas[str(target.get_name())]
            source_area = success_board.areas[str(source.get_name())]
            target_area.set_owner(self.player_name)
            target_area.set_dice(source.get_dice()-1)
            source_area.set_dice(1)
            """
                Nespesny utok -- aktualizacia boardu (podla pravidiel hry).
            """
            fail_board = copy.deepcopy(self.board)
            fsource_area = fail_board.areas[str(source.get_name())]
            fsource_area.set_dice(1)
            '''with open('fail' + str(self.counter) + '.save', 'wb') as f:
                save_state(f, fail_board, self.player_name, self.players_order)'''
            self.counter = self.counter + 1  

            """
                Vypocet poctov hracov nazive + hraca, ktory je aktualne na tahu. Podla toho urcena
                hlbka expectiminimax.
            """
            player_index = (self.players_order.index(self.player_name) + 1) % len(self.players_order)
            player = self.players_order[player_index]
            nb_players = self.board.nb_players_alive() - 1
       

            success_value = 0.0
            fail_value = 0.0
            '''
                Pravdepodobnost vyhry velka -- fail sa nezanoruje, ale hned vyhodnoti.
            '''
            if (atk_prob >= 0.8):        
                success_value = self.expectiminimax(success_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left) 
                fail_value = self.eval(fail_board)
            elif(atk_prob <= 0.47):
                '''
                '''
            else:
                success_value = self.expectiminimax(success_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left)
                fail_value = self.expectiminimax(fail_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left)

            """
                                       actuall_atack
                                        //     \\
                                   success      fail
                         atack(success_board)   atack(fail_board)
                             //                     \\
                             ...                    ...
             Z possible atacks sa vyberie jeden -- actual_atack. Spocita sa
             pst utoku atk_prob vytvori sa success_board pre uspesny utok a fail_board pre neuspesny.
             Zavola sa expectiminimax s kazdym z tychto boardov a po vrateni hodnot sa spocita vazeny
             priemer kde vaha je pst uspechu.

            """
            if (atk_prob > 0.47):
                '''
                    Prerezavanie.
                '''
                value = (atk_prob * success_value) + ((1-atk_prob) * fail_value)
                turns.append([source.get_name(), target.get_name(), [value, success_value, fail_value]])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)
    
    def expectiminimax(self, board, depth, player, level, ai, nb_moves_this_turn, nb_turns_this_game, time_left):
        value = 0
        """
            Listova uroven, ohodnotenie uzlu.
        """
        if(depth == 0):
            eval_value = self.eval(board)
            return (eval_value)
        """
            Uroven, na ktorej sa vytvara novy hrac (uroven protivnika). 
        """
        if(level == 1):
            ai_agent = AIstei(player, board, self.players_order)
        
            value = self.expectiminimax(board, depth, player, 2, ai_agent, nb_moves_this_turn, nb_turns_this_game, time_left)
        elif(level == 2): 
            """
                Uroven vsetkych tahov jedneho hraca. V command hodnota prikazu, ktory vyberie agent (stei).
            """    
            command = ai.ai_turn(board, nb_moves_this_turn, nb_turns_this_game, time_left)

            if isinstance(command, BattleCommand):
                """
                    Opatovne vytvorenie boardov a rekurzivne volanie expectiminimax s urovnou 2.
                """
                source = board.areas[str(command.source_name)]
                target = board.areas[str(command.target_name)]

                success_board = copy.deepcopy(board)
                target_area = success_board.areas[str(command.target_name)]
                source_area = success_board.areas[str(command.source_name)]
                target_area.set_owner(player)
                target_area.set_dice(source.get_dice()-1)
                source_area.set_dice(1)
                
                fail_board = copy.deepcopy(board)
                fsource_area = fail_board.areas[str(command.source_name)]
                fsource_area.set_dice(1)
                self.counter = self.counter + 1  

                atk_prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
                
                success_value = 0.0
                fail_value = 0.0

                if (atk_prob >= 0.8):        
                    success_value = self.expectiminimax(success_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left) 
                    fail_value = self.eval(fail_board)
                elif(atk_prob <= 0.47):
                    fail_value = self.eval(fail_board)
                    success_value = self.eval(success_board)
                else:
                    success_value = self.expectiminimax(success_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left)
                    fail_value = self.expectiminimax(fail_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left)

                value = (atk_prob * success_value) + ((1-atk_prob) * fail_value)
            elif isinstance(command, EndTurnCommand):
                """
                    Hrac ukoncil svoj tah -- vypocet dalsieho hraca na rade a pokracovanie.
                """
                players_index = (self.players_order.index(player) + 1) % len(self.players_order)
                players_count = self.players_order[players_index]
                value = self.expectiminimax(board, depth-1, players_count, 1, ai, nb_moves_this_turn, nb_turns_this_game, time_left)
                      
        return value

    '''
        Funkcia eval, ktora pocita s 2 dalsimi vlastnostami okrem skore.
    '''
    def eval(self, board):
        score = self.get_largest_region(board)
        cnt_not_attacking_boarder_areas = self.get_not_attacking_boarder_areas(board)
        cnt_attacking_boarder_dices = self.get_cnt_of_attacking_dices(board)
        eval_value = (score - 1 + 1/(cnt_not_attacking_boarder_areas + cnt_attacking_boarder_dices + 1))
        return eval_value

    def get_largest_region(self, board):
        self.largest_regions = []
        players_regions = board.get_players_regions(self.player_name)
        max_region_size = max(len(region) for region in players_regions)
        max_sized_regions = [region for region in players_regions if len(region) == max_region_size]

        for region in max_sized_regions:
            largest_region = []
            for area in region:
                largest_region.append(area)
            self.largest_regions.append(largest_region)
        return max_region_size

    '''
        Vrati cislo jednotkovych hranicnych oblasti, ktore nemozu utocit.
    '''
    def get_not_attacking_boarder_areas(self, board):
        cnt_not_attacking_boarder_areas = 0
        border_areas = board.get_player_border(self.player_name)
        for area in border_areas:
            if not area.can_attack():
                cnt_not_attacking_boarder_areas = cnt_not_attacking_boarder_areas + 1
        return cnt_not_attacking_boarder_areas

    '''
        Vrati pocet kociek, o ktore ma super viac na hraniciach a moze nimi na nas utocit.
    '''
    def get_cnt_of_attacking_dices(self, board):
        cnt_attacking_boarder_dices = 0
        border_areas = board.get_player_border(self.player_name)
        for area in border_areas:
            cnt_area_dices = area.get_dice()
            neighbours = area.get_adjacent_areas()
            for adj in neighbours:
                adjacent_area = board.get_area(adj)
                if adjacent_area.get_owner_name() != self.player_name:
                    cnt_neighbour_area_dices = adjacent_area.get_dice()
                    if cnt_neighbour_area_dices > cnt_area_dices:
                        cnt_attacking_boarder_dices = cnt_attacking_boarder_dices + cnt_neighbour_area_dices - cnt_area_dices
        return cnt_attacking_boarder_dices