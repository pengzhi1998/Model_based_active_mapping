import os
import numpy as np
num_landmarks = [3, 5, 7]
horizon = [9, 15, 21]
bound = [6, 10, 14]

def generate_testing_data():
    element_store_static = []
    element_store_moving = []

    for i in range(len(num_landmarks)):
        for num_tests in range(30):
            lx = np.random.uniform(low=-bound[i], high=bound[i], size=(num_landmarks[i], 1))
            ly = np.random.uniform(low=-bound[i], high=bound[i], size=(num_landmarks[i], 1))
            landmarks = np.concatenate((lx, ly), 1).reshape(num_landmarks[i] * 2, 1).tolist()
            agent_pos = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2),
                                  np.random.uniform(-np.pi, np.pi)]).tolist()
            U = np.zeros((num_landmarks[i] * 2, 1)).tolist()
            element_store_static.append([landmarks, agent_pos, U])

    for i in range(len(num_landmarks)):
        for num_tests in range(30):
            lx = np.random.uniform(low=-bound[i], high=bound[i], size=(num_landmarks[i], 1))
            ly = np.random.uniform(low=-bound[i], high=bound[i], size=(num_landmarks[i], 1))
            landmarks = np.concatenate((lx, ly), 1).reshape(num_landmarks[i] * 2, 1).tolist()
            agent_pos = np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2),
                                  np.random.uniform(-np.pi, np.pi)]).tolist()
            U = np.array(np.random.uniform(-.8, .8, size=(2,)).tolist() * num_landmarks[i]).\
                reshape(num_landmarks[i] * 2, 1).tolist()
            element_store_moving.append([landmarks, agent_pos, U])

    return element_store_static, element_store_moving


if __name__ == '__main__':
    generate_testing_data_ = generate_testing_data()
    print(generate_testing_data_[0], "\n",
          generate_testing_data_[1])
    # generated_date_static =
    # generated_date_moving =
