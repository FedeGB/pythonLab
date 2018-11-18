from laberint_game import LaberintGame
from random import randint
import numpy as np
import tflearn
import math
from tflearn.layers.core import input_data, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean
from collections import Counter

class LabyrinthNN:
    def __init__(self, initial_games = 10000, test_games = 1000, goal_steps = 2000, lr = 1e-2, filename = 'lab_nn.tflearn'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.filename = filename
        self.avatarX = 0
        self.avatarY = 0
        self.exitX = 0
        self.exitY = 0
        # self.vectors_and_keys = [
        #         [[-1, 0], 0],
        #         [[0, 1], 1],
        #         [[1, 0], 2],
        #         [[0, -1], 3]
        #         ]

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = LaberintGame()
            _, done, prev_score, board = game.start()
            init_exit_position(board)
            update_avatar_position(board)
            prev_observation = self.generate_observation(board)
            prev_exit_distance = self.get_exit_distance()
            for _ in range(self.goal_steps):
                game_action = self.generate_action()
                done, score, board  = game.step(game_action)
                update_avatar_position(board)
                if done:
                    training_data.append([self.add_action_to_observation(prev_observation, game_action), -1])
                    break
                else:
                    exit_distance = self.get_exit_distance()
                    if score > prev_score or exit_distance < prev_exit_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, game_action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, game_action), 0])
                    prev_observation = self.generate_observation(board)
                    prev_exit_distance = exit_distance
        return training_data

    def generate_action(self):
        action = randint(0,3)
        return action

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

    def generate_observation(self, board):
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
        return np.array([obstacle_left, obstacle_up, obstacle_right, obstacle_down])

    def add_action_to_observation(self, observation, action):
        return np.append([action], observation)

    def get_snake_direction_vector(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_exit_direction_vector(self):
        return np.array([self.exitX, self.exitY]) - np.array([self.avatarX, self.avatarY])

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def get_exit_distance(self):
        return np.linalg.norm(self.get_exit_direction_vector())

    def is_obstacle_on_direction(self, board, position):
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

        if(board[localx][localy] == '#'):
            return 1 # wall 
        elif(board[localx][localy] == 't'):
            return -1 # trap
        return 0

    def is_direction_blocked(self, snake, direction):
        point = np.array(snake[0]) + np.array(direction)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def turn_vector_to_the_left(self, vector):
        return np.array([-vector[1], vector[0]])

    def turn_vector_to_the_right(self, vector):
        return np.array([vector[1], -vector[0]])

    def get_angle(self, a, b):
        a = self.normalize_vector(a)
        b = self.normalize_vector(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def model(self):
        network = input_data(shape=[None, 5, 1], name='input')
        network = fully_connected(network, 25, activation='relu')
        network = fully_connected(network, 1, activation='linear')
        network = regression(network, optimizer='adam', learning_rate=self.lr, loss='mean_square', name='target')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5, 1)
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
            game = LaberintGame()
            _, score, snake, food = game.start()
            prev_observation = self.generate_observation(snake, food)
            for _ in range(self.goal_steps):
                predictions = []
                for action in range(-1, 2):
                   predictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
                action = np.argmax(np.array(predictions))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food  = game.step(game_action)
                game_memory.append([prev_observation, action])
                if done:
                    print('-----')
                    print(steps)
                    print(snake)
                    print(food)
                    print(prev_observation)
                    print(predictions)
                    break
                else:
                    prev_observation = self.generate_observation(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        print('Average steps:',mean(steps_arr))
        print(Counter(steps_arr))
        print('Average score:',mean(scores_arr))
        print(Counter(scores_arr))

    def visualise_game(self, model):
        game = LaberintGame(gui = True)
        _, _, snake, food = game.start()
        prev_observation = self.generate_observation(snake, food)
        for _ in range(self.goal_steps):
            precictions = []
            for action in range(-1, 2):
               precictions.append(model.predict(self.add_action_to_observation(prev_observation, action).reshape(-1, 5, 1)))
            action = np.argmax(np.array(precictions))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food  = game.step(game_action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food)

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
    LabyrinthNN().train()