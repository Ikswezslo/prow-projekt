import subprocess
from turtle import width
import numpy as np
import matplotlib.pyplot as plt

def load_inst(src="../bin/inst.txt"):
    f = open(src,"r")
    data = f.read().split()
    n = int(data[0])
    data = data[1:]
    data = list(map(float, data))
    px = np.array([data[1::3]]).T
    py = np.array([data[2::3]]).T
    return px,py,n

def gen_inst(n, filename = "../bin/inst.txt"):
    max_x = 100
    max_y = 100
    px = np.random.rand(n,1) * max_x
    py = np.random.rand(n,1) * max_y
    f = open(filename, "w")
    f.write(str(n) + "\n")
    for i in range(n):
        f.write(str(i)+" "+str(round(px[i][0],3))+" "+str(round(py[i][0],3))+"\n")
    f.close()
    px = [round(px[i][0],3) for i in range(n)]
    py = [round(py[i][0],3) for i in range(n)]
    return px,py

def calculate_solution(px, py, solution, debug=False):
    n = len(solution)
    result = 0

    if len(solution) != len(set(solution)):
        print("Błąd!!!")

    for i in range(n):
        a = solution[i]
        b = solution[(i+1) % n]
        result += ((px[a]-px[b])**2 + (py[a]-py[b])**2)**(0.5)

    if debug:
        print(result)
    return result

def change_init(ant_number=10,iteration_number=100,pher_start_value=0.1,pher_evap=0.5,pher_infl=1.5,length_infl=5,const_Q=100, num_threads=1):
    f = open("../bin/aco_config.txt", "w")
    f.write("ant_number "+str(int(ant_number))+"\n")
    f.write("iteration_number "+str(int(iteration_number))+"\n")
    f.write("pheromone_start_value "+str(pher_start_value)+"\n")
    f.write("pheromone_evaporation "+str(pher_evap)+"\n")
    f.write("pheromone_influence "+str(pher_infl)+"\n")
    f.write("length_influence "+str(length_infl)+"\n")
    f.write("const_Q "+str(const_Q)+"\n")
    f.write("num_threads "+str(num_threads)+"\n")
    f.close()

def do_aco(print_output=False, program = "aco_seq", src="../bin/inst.txt"):
    f = open(src,"r")
    f.close()
    
    proc = subprocess.run(['../bin/./'+program],capture_output=True, cwd="../bin")
    result = proc.stdout.splitlines()
    time = result[0].decode("utf-8").split()[1]
    cost = result[1].decode("utf-8").split()[1]
    solution = list(map(int,result[2].decode("utf-8").split()[1:]))
    if print_output:
        print("Czas: ", time)
        print("Wynik: ", cost)
        print("Rozwiązanie: ", solution)
    return time,cost,solution

def plot_solution(px,py,solution):
    plt.scatter(px,py, c='black', s=50)
    x = []
    y = []
    for i in solution:
        x.append(px[i])
        y.append(py[i])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x,y, linewidth=2)
    plt.show()

def main():
    px,py,n = load_inst("inst.txt")
    change_init(ant_number=n/3,iteration_number=200, num_threads=8)
    time,cost,solution = do_aco(print_output=True, src="inst.txt")
    plot_solution(px,py,solution)

if __name__ == '__main__':
    main()

