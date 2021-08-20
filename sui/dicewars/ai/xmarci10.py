import random
import copy
import logging

from dicewars.client.game.board import Board
from dicewars.ai.utils import possible_attacks, attack_succcess_probability, save_state
from dicewars.ai.utils import probability_of_holding_area, probability_of_successful_attack

from dicewars.client.ai_driver import BattleCommand, EndTurnCommand, AIDriver
from dicewars.ai.dt.stei import AI as AIstei
from dicewars.ai.dt.wpm_c import AI as AIwpmc

class AI:
    """Naive player agent

    This agent performs all possible moves in random order
    """

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
        self.target = 0
        # with open('debug.save', 'wb') as f:
        #     save_state(f, board, player_name, players_order)

    def ai_turn(self, board, nb_moves_this_turn, nb_turns_this_game, time_left):
        """AI agent's turn

        Agent gets a list preferred moves and makes such move that has the
        highest estimated hold probability. If there is no such move, the agent
        ends it's turn.
        """
        self.board = board
        self.logger.debug("Looking for possible turns.")
        
        turns = self.possible_turns(nb_moves_this_turn, nb_turns_this_game, time_left)

        self.logger.debug("<>turns = {}".format(turns))
        if turns:
            turn = turns[0]
            area_name = turn[0]
            self.logger.debug("Possible turn: {}".format(turn))
            if(turn[2] >= 0.5):
                return BattleCommand(area_name, turn[1])

        self.logger.debug("No more plays.")
        return EndTurnCommand()

    def possible_turns(self, nb_moves_this_turn, nb_turns_this_game, time_left):
        turns = []
        for source, target in possible_attacks(self.board, self.player_name):
            self.target = target.get_name()
            atk_prob = probability_of_successful_attack(self.board, source.get_name(), target.get_name())
            if(atk_prob < 0.47):
                continue
            # if succesfull attack
            success_board = copy.deepcopy(self.board)
            target_area = success_board.areas[str(target.get_name())]
            source_area = success_board.areas[str(source.get_name())]
            target_area.set_owner(self.player_name)
            target_area.set_dice(source.get_dice()-1)
            source_area.set_dice(1)
            # with open('success' + str(self.counter) + '.save', 'wb') as f:
            #     save_state(f, success_board, self.player_name, self.players_order)

            # if unsuccesfull attack
            fail_board = copy.deepcopy(self.board)
            fsource_area = fail_board.areas[str(source.get_name())]
            fsource_area.set_dice(1)
            # with open('fail' + str(self.counter) + '.save', 'wb') as f:
            #     save_state(f, fail_board, self.player_name, self.players_order)
            self.counter = self.counter + 1  

            player_index = (self.players_order.index(self.player_name) + 1) % len(self.players_order)
            player = self.players_order[player_index]
            nb_players = self.board.nb_players_alive() - 1
       
            success_value = 0.0
            fail_value = 0.0

            self.logger.debug("{0}->{1}".format(source.get_name(), target.get_name()))
            self.logger.debug("####atk_prob = {}".format(atk_prob))
            # if (atk_prob >= 0.8):        
            #     success_value = self.expectiminimax(success_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left) 
            #     self.logger.debug("####success_value = {}".format(success_value))
            #     self.logger.debug("####fail_value = {}".format(fail_value))
            # elif(atk_prob <= 0.2):
            #     self.logger.debug("####success_value = {}".format(success_value))
            #     fail_value = self.expectiminimax(fail_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left)
            #     self.logger.debug("####fail_value = {}".format(fail_value))
            # else:
            success_value = self.expectiminimax(success_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left) 
            self.logger.debug("####success_value = {}".format(success_value))
            # fail_value = self.expectiminimax(fail_board, nb_players, player, 1, None, nb_moves_this_turn, nb_turns_this_game, time_left)                    
            # self.logger.debug("####fail_value = {}".format(fail_value))

            value = (atk_prob * success_value) #+ ((1-atk_prob) * fail_value)        

            self.logger.debug("value = {}".format(value))
            turns.append([source.get_name(), target.get_name(), value])

            if(value >= 0.8):
                break

        return sorted(turns, key=lambda turn: turn[2], reverse=True)
    
    def expectiminimax(self, board, depth, player, level, ai, nb_moves_this_turn, nb_turns_this_game, time_left):
        value = 0

        self.logger.debug("<<<<<<<<depth: {}".format(depth))
        self.logger.debug("<<<<<<<<player_name: {}".format(player))
        if(depth == 0):
            self.logger.debug("<<<<<<<<<<<<debug{}".format(self.counter))
            self.logger.debug("<<<<<<<<largest_region: {}".format((self.get_largest_region(board))))
            # return (self.get_largest_region(board))
            return 1
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
                self.logger.debug(">>>printing success board({})".format(self.counter))
                # with open('success' + str(self.counter) + '.save', 'wb') as f:
                #     save_state(f, success_board, player, self.players_order)
                

                fail_board = copy.deepcopy(board)
                fsource_area = fail_board.areas[str(command.source_name)]
                fsource_area.set_dice(1)
                self.logger.debug(">>>printing fail board({})".format(self.counter))
                # with open('fail' + str(self.counter) + '.save', 'wb') as f:
                #     save_state(f, fail_board, player, self.players_order)  
                self.counter = self.counter + 1  

                atk_prob = probability_of_successful_attack(board, source.get_name(), target.get_name())
                self.logger.debug("####atk_prob = {}".format(atk_prob))
                
                success_value = 0.0
                fail_value = 0.0

                # if (atk_prob >= 0.8):        
                #     success_value = self.expectiminimax(success_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left) 
                #     self.logger.debug("####1success_value = {}".format(success_value))
                #     self.logger.debug("####1fail_value = {}".format(fail_value))
                # elif(atk_prob <= 0.2):
                #     self.logger.debug("####2success_value = {}".format(success_value))
                #     fail_value = self.expectiminimax(fail_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left)
                #     self.logger.debug("####2fail_value = {}".format(fail_value))
                # else:
                if(self.target == command.target_name):
                    success_value = atk_prob
                    fail_value = 1 - atk_prob                 
                else: 
                    if (atk_prob >= 0.8):    
                        success_value = self.expectiminimax(success_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left) 
                        self.logger.debug("####success_value = {}".format(success_value))
                        fail_value = 1 - atk_prob
                    elif(atk_prob <= 0.2):
                        success_value = atk_prob
                        fail_value = self.expectiminimax(fail_board, depth, player, 2, ai, nb_moves_this_turn, nb_turns_this_game, time_left)                    
                        self.logger.debug("####3fail_value = {}".format(fail_value))

                value = (atk_prob * success_value) + ((1-atk_prob) * fail_value)
                self.logger.debug("####atk_prob = {}".format(atk_prob))
                self.logger.debug("####value = {}".format(value))
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
