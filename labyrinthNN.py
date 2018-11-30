from laberint_game import LaberintGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter
import manual

class LabyrinthNN:
    def __init__(self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'lab_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.goal_steps_train = 500
        self.lr = lr
        self.filename = filename
        self.avatarX = 0
        self.avatarY = 0
        self.exitX = 0
        self.exitY = 0
        self.positionDic = {}
        self.manualGames = 1
        self.maxBoardDistance = 20.0
        # self.vectors_and_keys = [
        #         [[-1, 0], 0],
        #         [[0, 1], 1],
        #         [[1, 0], 2],
        #         [[0, -1], 3]
        #         ]

    def initial_population(self):
        training_data = []
        for number in range(self.initial_games):
            if number < self.manualGames:
                game = LaberintGame(gui = True)
            else:
                game = LaberintGame()
            self.positionDic.clear()
            # print("New game started")
            done, prev_score, board = game.start()
            self.maxBoardDistance = self.get_max_board_distance(board)
            self.init_exit_position(board)
            self.update_avatar_position(board)
            # prev_observation = self.generate_observation(board)
            prev_exit_distance = self.get_exit_distance()
            for _ in range(self.goal_steps * 10):
                if number < self.manualGames:
                    game_action = manual.get()
                else:
                    game_action = self.generate_action()
                prev_observation = self.generate_observation(board, prev_score)
                done, score, board  = game.step(game_action)
                self.update_avatar_position(board)
                # print(str(self.avatarX) + "," + str(self.avatarY))
                # print(str(self.exitX) + "," + str(self.exitY))
                if done:
                    if score < prev_score:
                        # print("died")
                        training_data.append([self.add_action_to_observation(prev_observation, game_action), -1])
                    else:
                        print("win")
                        training_data.append([self.add_action_to_observation(prev_observation, game_action), 1])
                    break
                else:
                    exit_distance = self.get_exit_distance()
                    if (score + 2 >= prev_score):
                        # print("good direction")
                        training_data.append([self.add_action_to_observation(prev_observation, game_action), 1])
                    else:
                        # print("bad direction")
                        training_data.append([self.add_action_to_observation(prev_observation, game_action), 0])
                    prev_exit_distance = exit_distance
                    prev_score = score
        return training_data

    def generate_action(self):
        action = randint(0,3)
        posStr = self.get_position_string(self.avatarX, self.avatarY)
        if posStr in self.positionDic:
            actions = self.positionDic[posStr]
            if(randint(0,100) % 2 == 0):
                return actions[randint(0,len(actions)-1)]
            if len(actions) == 4:
                return action
            while action in actions:
                action = randint(0,3)
            actions.append(action)
            self.positionDic[posStr] = actions
        else:
            actions = []
            actions.append(action)
            self.positionDic[posStr] = actions
        return action

    def get_position_string(self, posX, posY):
        return str(posX) + "," + str(posY)

    def update_avatar_position(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(board[i][j] == 'A'):
                    self.avatarX = i
                    self.avatarY = j
                    return

    def init_exit_position(self, board):
        for i in range(len(board)):
            for j in range(len(board[i])):
                if(board[i][j] == 'x'):
                    self.exitX = i
                    self.exitY = j
                    return        

    def generate_observation(self, board, score):
        # snake_direction = self.get_snake_direction_vector(snake)
        # food_direction = self.get_food_direction_vector(snake, food)
        obstacle_left = self.is_obstacle_on_direction(board, 0)
        obstacle_down = self.is_obstacle_on_direction(board, 1)
        obstacle_right = self.is_obstacle_on_direction(board, 2)
        obstacle_up = self.is_obstacle_on_direction(board, 3)
        # barrier_left = self.is_direction_blocked(snake, self.turn_vector_to_the_left(snake_direction))
        # barrier_front = self.is_direction_blocked(snake, snake_direction)
        # barrier_right = self.is_direction_blocked(snake, self.turn_vector_to_the_right(snake_direction))
        # angle = self.get_angle(snake_direction, food_direction)
        board_width = len(board) - 1
        board_height = len(board[0]) - 1
        distance_value = (self.get_exit_distance()) / self.maxBoardDistance
        return np.array([obstacle_left, obstacle_up, obstacle_right, obstacle_down, 
            self.avatarX / board_width, self.avatarY / board_height, distance_value])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    # def get_snake_direction_vector(self, snake):
    #     return np.array(snake[0]) - np.array(snake[1])

    def get_exit_direction_vector(self):
        return np.array([self.exitX, self.exitY]) - np.array([self.avatarX, self.avatarY])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_exit_distance(self):
        return np.linalg.norm(self.get_exit_direction_vector())

    def get_max_board_distance(self, board):
        distance_vector = np.array([len(board) - 1, len(board[0]) - 1]) - np.array([0, 0])
        return np.linalg.norm(distance_vector)        

    def is_obstacle_on_direction(self, board, position):
        value = self.get_value_on_direction(board, position)
        if(value == 'w'):
            return 1 # wall 
        elif(value == 't'):
            return -1 # trap
        elif(value == 'v'):
            return -0.5 # already visited
        return 0

    def get_value_on_direction(self, board, position):
        localx = self.avatarX
        localy = self.avatarY
        if position == 0:
            localy = self.avatarY-1
            # LEFT
        elif position == 1:
            localx = self.avatarX+1
            # DOWN
        elif position == 2:
            localy = self.avatarY+1
            # RIGHT
        elif position == 3:
            localx = self.avatarX-1
            # UP
        board_width = len(board)
        board_height = len(board[0])
        if(localx >= board_width or localy >= board_height or localx <= 0 or localy <= 0):
            return 'w'
        posStr = self.get_position_string(localx, localy)
        if posStr in self.positionDic:
            return 'v'
        return board[localx][localy]

    def model(self):
        network = input_data(shape=[None, 8, 1], name='input')
        network = fully_connected(network, 250, activation='relu')
        network = fully_connected(network, 250, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 8, 1)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X,y, n_epoch = 3, shuffle = True, run_id = self.filename)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_memory = []
            game = LaberintGame(gui = False)
            done, score, board = game.start()
            self.init_exit_position(board)
            self.update_avatar_position(board)
            self.positionDic.clear()
            prev_observation = self.generate_observation(board, score)
            # print("new game")
            for _ in range(self.goal_steps_train):
                predictions = []
                for action in range(0, 4):
                    predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 8, 1)))
                action = np.argmax(np.array(predictions))
                # game_action = self.get_game_action(snake, action - 1)
                self.positionDic[self.get_position_string(self.avatarX, self.avatarY)] = 1
                done, score, board  = game.step(action)
                self.update_avatar_position(board)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(board)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(board, score)
                    steps += 1
            # print("End")
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = LaberintGame(gui = True)
        _, _, board = game.start()
        self.init_exit_position(board)
        self.update_avatar_position(board)
        prev_observation = self.generate_observation(board)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(0, 4):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 8, 1)))
            action = np.argmax(np.array(precictions))
            # game_action = self.get_game_action(snake, action - 1)
            done, _, board  = game.step(action)
            self.update_avatar_position(board)
            if done:
                break
            else:
                prev_observation = self.generate_observation(board)

    def train(self):
        training_data = self.initial_population()
        nn_model = self.model()
        nn_model = self.train_model(training_data, nn_model)
        self.test_model(nn_model)

    def visualise(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.visualise_game(nn_model)

    def test(self):
        nn_model = self.model()
        nn_model.load(self.filename)
        self.test_model(nn_model)

if __name__ == "__main__":
    LabyrinthNN(30000, 1000, 50000).train()