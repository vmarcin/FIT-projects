import random
import copy
import logging

from dicewars.client.game.board import Board
from dicewars.client.game.area import Area
from dicewars.ai.utils import possible_attacks, attack_succcess_probability, save_state
from dicewars.ai.utils import probability_of_holding_area, probability_of_successful_attack

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, AIDriver
from dicewars.ai.dt.stei import AI as AIstei
from dicewars.ai.dt.wpm_c import AI as AIwpmc

class AI:

    def __init__(self, player_name, board, players_order):
        """
        Parameters
        ----------
        game : Game
        """
        self.player_name = player_name
        self.logger = logging.getLogger('AI')
        self.players_order = players_order
        self.counter = 1
        self.board = board

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Agent gets a list preferred moves and makes such move that has the
        highest estimated hold probability. If there is no such move, the agent
        ends it's turn.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        
        turns = self.possible_turns(nb_moves_this_turn, nb_turns_this_game, time_left)

        if turns:
            turn = turns[0]
            area_name = turn[0]
            self.logger.debug("Possible turn: {}".format(turn))
            value = turn[2][0]
            self.logger.debug("hodnota {}".format(value))
            self.logger.debug("UTOK")
            return BattleCommand(area_name, turn[1])

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self, nb_moves_this_turn, nb_turns_this_game, time_left):
        turns = []

        for source, target in possible_attacks(self.board, self.player_name):
            atk_prob = probability_of_successful_attack(self.board, source.get_name(), target.get_name())

            if(atk_prob <= 0.47):
                continue
            # if succesfull attack
            success_board = copy.deepcopy(self.board)
            target_area = success_board.areas[str(target.get_name())]
            source_area = success_board.areas[str(source.get_name())]
            target_area.set_owner(self.player_name)
            target_area.set_dice(source.get_dice()-1)
            source_area.set_dice(1)

            # if unsuccesfull attack
            fail_board = copy.deepcopy(self.board)
            fsource_area = fail_board.areas[str(source.get_name())]
            fsource_area.set_dice(1)

            self.counter = self.counter + 1  

            player_index = (self.players_order.index(self.player_name) + 1) % len(self.players_order)
            player = self.players_order[player_index]
            nb_players = self.board.nb_players_alive() - 1

            success_value = 0.0
            fail_value = 0.0

            if (atk_prob >= 0.8):        
                success_value = self.expectiminimax(success_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left) 
                fail_value = self.eval_immediately(fail_board)
            else:
                success_value = self.expectiminimax(success_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left) 
                fail_value = self.expectiminimax(fail_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left)                    

            
            value = (atk_prob * success_value) + ((1-atk_prob) * fail_value)        
            self.logger.debug("value = {}".format(value))
            turns.append([source.get_name(), target.get_name(), [value, success_value, fail_value]])

        return sorted(turns, key=lambda turn: turn[2], reverse=True)
    
    def expectiminimax(self, board, depth, player, level, ai, nb_moves_this_turn, nb_turns_this_game, time_left):
        value = 0

        if(depth == 0):
            eval_value = self.eval_immediately(board)
            return (eval_value)

        if(level == 1):
            ai_agent = AIstei(player, board, self.players_order)
        
            value = self.expectiminimax(board, depth, player, 2, ai_agent, nb_moves_this_turn, nb_turns_this_game, time_left)
        elif(level == 2): 
            command = ai.ai_turn(board, nb_moves_this_turn, nb_turns_this_game, time_left)
  
            if isinstance(command, BattleCommand):
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

                atk_prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
                
                success_value = 0.0
                fail_value = 0.0

                if (atk_prob >= 0.8):        
                    success_value = self.expectiminimax(success_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left) 
                    fail_value = self.eval_immediately(fail_board) 
                elif(atk_prob <= 0.47):
                    fail_value = self.eval_immediately(fail_board)
                    success_value = self.eval_immediately(success_board)
                else:
                    success_value = self.expectiminimax(success_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left) 
                    fail_value = self.expectiminimax(fail_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left)                    

                value = (atk_prob * success_value) + ((1-atk_prob) * fail_value)
            elif isinstance(command, EndTurnCommand):
                players_index = (self.players_order.index(player) + 1) % len(self.players_order)
                players_count = self.players_order[players_index]
                value = self.expectiminimax(board, depth-1, players_count, 1, ai, nb_moves_this_turn, nb_turns_this_game, time_left)
                      
        return value

    def probability_of_holding_source(self, source_name, area_dice, player_name, target_area):
        source_area = self.board.get_area(source_name)
        probability = 1.0
        for adj in source_area.get_adjacent_areas():
            adjacent_area = self.board.get_area(adj)
            if(adjacent_area == target_area):
                continue
            elif(adjacent_area.get_owner_name() != player_name):
                enemy_dice = adjacent_area.get_dice()
                if enemy_dice == 1:
                    continue
                self.logger.debug("<<<<adj = {}".format(adjacent_area.get_name()))
                lose_prob = attack_succcess_probability(enemy_dice, area_dice)
                hold_prob = 1.0 - lose_prob
                probability *= hold_prob
        return probability  

    '''funkcia eval, ktora pocita s 2 dalsimi vlastnostami okrem skore, momentalne nepouzivana'''
    def eval_immediately(self, board):
        score = self.get_largest_region(board)
        cnt_not_attacking_boarder_areas = self.get_not_attacking_boarder_areas(board)
        cnt_attacking_boarder_dices = self.get_cnt_of_attacking_dices(board)
        eval_value = (score - 1 + 1/(cnt_not_attacking_boarder_areas + cnt_attacking_boarder_dices + 1))
        return eval_value

    def eval(self, board):
        score = self.get_largest_region(board)
        return score 

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

    '''vrati cislo jednotkovych hranicnych oblasti, ktore nemozu utocit -- chceme ho co najmensie'''
    def get_not_attacking_boarder_areas(self, board):
        cnt_not_attacking_boarder_areas = 0
        border_areas = board.get_player_border(self.player_name)
        for area in border_areas:
            if not area.can_attack():
                cnt_not_attacking_boarder_areas = cnt_not_attacking_boarder_areas + 1
        return cnt_not_attacking_boarder_areas

    '''vrati pocet kociek, o ktore ma super viac na hraniciach a moze nimi na nas utocit -- chceme ho co najmensie'''
    def get_cnt_of_attacking_dices(self, board):
        cnt_attacking_boarder_dices = 0
        border_areas = board.get_player_border(self.player_name)
        for area in border_areas:
            cnt_area_dices = area.get_dice()
            '''self.logger.debug("<<<<area_dices = {}".format(cnt_area_dices))'''
            neighbours = area.get_adjacent_areas()
            for adj in neighbours:
                adjacent_area = board.get_area(adj)
                if adjacent_area.get_owner_name() != self.player_name:
                    cnt_neighbour_area_dices = adjacent_area.get_dice()
                    '''self.logger.debug("<<<<neighbour_najdeny, ma {} kociek".format(cnt_neighbour_area_dices))'''
                    if cnt_neighbour_area_dices > cnt_area_dices:
                        cnt_attacking_boarder_dices = cnt_attacking_boarder_dices + cnt_neighbour_area_dices - cnt_area_dices
                        '''self.logger.debug("<<<<rozdiel od neighboura je {} kociek".format(cnt_neighbour_area_dices - cnt_area_dices))'''
        return cnt_attacking_boarder_dices