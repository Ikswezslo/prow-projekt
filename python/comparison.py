import aco
import pandas as pd


def get_corrected_value(time_list: list):
    return (sum(time_list) - min(time_list) - max(time_list)) / (len(time_list) - 2)

repeats = 7
num_cities_list = list(range(100,601,50))

col_seq = []
col_omp = []
col_cuda_v1 = []
col_cuda_v2 = []
for n in num_cities_list:
    
    time_list_seq = []
    time_list_omp = []
    time_list_cuda_v1 = []
    time_list_cuda_v2 = []
    for i in range(repeats):
        aco.gen_inst(n)
        aco.change_init(ant_number=n/3, num_threads=8)
        
        time,cost,solution = aco.do_aco(print_output=False, program="aco_seq")
        time_list_seq.append(float(time))
        time,cost,solution = aco.do_aco(print_output=False, program="aco_omp")
        time_list_omp.append(float(time))
        time,cost,solution = aco.do_aco(print_output=False, program="aco_cuda")
        time_list_cuda_v1.append(float(time))
        time,cost,solution = aco.do_aco(print_output=False, program="aco_cuda2")
        time_list_cuda_v2.append(float(time))

        
    value = get_corrected_value(time_list_seq)
    print(f"Seq:     \t Miasta: {n}\t czas całkowity: {round(sum(time_list_seq), 3)}\t czas skorygowany: {round(value, 3)}")
    col_seq.append(value)
    value = get_corrected_value(time_list_omp)
    print(f"Omp:     \t Miasta: {n}\t czas całkowity: {round(sum(time_list_omp), 3)}\t czas skorygowany: {round(value, 3)}")
    col_omp.append(value)
    value = get_corrected_value(time_list_cuda_v1)
    print(f"Cuda v1: \t Miasta: {n}\t czas całkowity: {round(sum(time_list_cuda_v1), 3)}\t czas skorygowany: {round(value, 3)}")
    col_cuda_v1.append(value)
    value = get_corrected_value(time_list_cuda_v2)
    print(f"Cuda v2: \t Miasta: {n}\t czas całkowity: {round(sum(time_list_cuda_v2), 3)}\t czas skorygowany: {round(value, 3)}")
    col_cuda_v2.append(value)


df = pd.DataFrame({"seq": col_seq, "omp": col_omp, "cuda_v1": col_cuda_v1, "cuda_v2": col_cuda_v2})
df.index = num_cities_list
df.to_csv("results/comparison.csv")