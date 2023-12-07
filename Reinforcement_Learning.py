import subprocess
import numpy as np
import time

start_time = time.time()
iteration = 1
max_iter = 100
good_dir = False
grad = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
while iteration < max_iter:

    # Simulated annealing to the nudge
    nudge_multiplier = 2 * (max_iter / iteration) / max_iter

    # Get parameters from text
    prev_params = np.genfromtxt('Parameters.txt', delimiter=',')
    np.savetxt("curr_best_params.txt", prev_params, delimiter=",")
    # random nudge value

    random_nudge = nudge_multiplier * np.random.uniform(-1, 1, (3, 3))

    if good_dir is True:
        nudge = grad
    else:
        nudge = random_nudge

    # execute random nudge
    new_params = prev_params + nudge

    # save params for the program
    np.savetxt("Parameters.txt", new_params, delimiter=",")

    # get previous result to compare to from RL_Logs.txt

    command = f"Get-Content {"RL_Logs.txt"} | Select-Object -Last 1"
    RL_output = subprocess.check_output(['powershell', '-Command', command], shell=True, text=True)
    previous_result = float(RL_output)
    # run the autorun
    subprocess.run(
        'python simulator.py --player_1 student_agent_3 --player_2 student_agent --autoplay --autoplay_runs 50')

    # get the new result from RL_Logs.txt
    RL_output = subprocess.check_output(['powershell', '-Command', command], shell=True, text=True)
    new_result = float(RL_output)

    print("Iteration: ", iteration, " complete!, new result: ", new_result, "Previous result: ", previous_result)

    # If the new result is not better, return to previous result and parameters
    if new_result < previous_result:
        np.savetxt("Parameters.txt", prev_params, delimiter=",")

        f = open("RL_Logs.txt", "a")
        f.write('\n')
        f.write("Reusing previous result: ")
        f.write('\n')
        f.write(str(previous_result))
        f.close()
    else:
        grad = new_params - prev_params

    iteration += 1
