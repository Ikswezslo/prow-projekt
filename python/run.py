import aco
n = 30
px, py = aco.gen_inst(n)
aco.change_init(ant_number=n/3, num_threads=8)
time, cost, solution = aco.do_aco(print_output=True, program="aco_cuda2")
aco.plot_solution(px, py, solution)


