import os
cwd1 = os.path.join(os.getcwd(), "../model_based_active_mapping/scripts/run_model_based_testing.py")
cwd2 = os.path.join(os.getcwd(), "../toy_active_mapping_ppo/agent_test.py")

results_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 "./testing_results.txt")
num_landmarks = [3, 5, 7]
horizon = [9, 15, 21]
seed = ['0', '10', '100']
model = ['model_based', 'model_free']
motion_model = [1, 2]  # 1 for static noisy model, 2 for moving model

def auto_test():


    for l in model:
        for mm in motion_model:
            count = 0
            for j in range(len(num_landmarks)):
                label = [l + "_" + "num_landmarks" + str(num_landmarks[j]) + "_" + str(mm), "\n"]
                my_open_results = open(results_file_path, "a")
                for element in label:
                    print("\n\n\n\n\n")
                    my_open_results.write(element)
                my_open_results.close()
                for k in seed:
                    if l == 'model_based':
                        os.system('python {} --seed={} --count={} --motion-model={} '
                                  '--for-comparison=1'.format(cwd1, k, count, str(mm)))
                    else:
                        os.system('python {} --seed={} --count={} --motion-model={} --num-landmarks={} '
                                  '--for-comparison=1'.format(cwd2, k, count, str(mm), str(num_landmarks[j])))
                    count += 1
                newline = ["\n"]
                my_open_results = open(results_file_path, "a")
                for element in newline:
                    my_open_results.write(element)
                my_open_results.close()
    # my_open_results.close()

if __name__ == '__main__':
    auto_test()
