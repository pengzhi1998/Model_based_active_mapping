import os
cwd = os.path.join(os.getcwd(), "./agent_test.py")
data_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                "plots/landmarks_agent_init_pos.txt")
results_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                "plots/testing_results.txt")
num_landmarks = [3, 5, 8]
horizon = [8, 15, 18]
bound = [8, 10, 12]
seed = ['0', '10', '100']
model = ['attention', 'mlp']

def auto_test():
    # my_open_results = open(results_file_path, "a")

    for l in model:
        count = 0
        for j in range(len(num_landmarks)):
            # label = ["model_" + l + "_" + "num_landmarks" + str(num_landmarks[j]), "\n"]
            # for element in label:
            #     my_open_results.write(element)
            for k in seed:
                os.system('python {} --num-landmarks={} --horizon={} --bound={} --seed={} --model={} --count={} --for-comparison=1'.format(cwd, num_landmarks[j], horizon[j], bound[j], k, l, count))
                count += 1
            # newline = ["\n"]
    #         for element in newline:
    #             my_open_results.write(element)
    # my_open_results.close()

if __name__ == '__main__':
    auto_test()
