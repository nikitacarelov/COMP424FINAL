import subprocess
import numpy as np
import time

start_time = time.time()
iteration = 1
max_iter = 20
while iteration < 20 or (time.time() - start_time) > 1000:

    # Simulated annealing to the nudge
    nudge_multiplier = 3 * (max_iter/iteration)/max_iter

    # Get parameters from text
    prev_params = np.genfromtxt('Parameters.txt', delimiter=',')

    # random nudge value
    random_nudge = nudge_multiplier * np.random.uniform(-1, 1, (3, 3))

    # execute random nudge
    new_params = prev_params + random_nudge

    # save params for the program
    np.savetxt("Parameters.txt", new_params, delimiter=",")

    # get previous result to compare to from RL_Logs.txt

    command = f"Get-Content {"RL_Logs.txt"} | Select-Object -Last 1"
    RL_output = subprocess.check_output(['powershell', '-Command', command], shell=True, text=True)
    previous_result = float(RL_output)
    # run the autorun
    subprocess.run('python simulator.py --player_1 student_agent_3 --player_2 student_agent  --autoplay --autoplay_runs 15')

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

    iteration += 1
