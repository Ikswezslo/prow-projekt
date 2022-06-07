import aco
import pandas as pd

repeats = 7
max_num_threads = 12
num_cities_list = list(range(100,301,50))


num_threads_list = list(range(max_num_threads+1))
columns = ['n'+str(x) for x in num_cities_list]

data = []
for num_threads in num_threads_list:
    row = []
    for n in num_cities_list:
        aco.gen_inst(n)
        time_list = []
        for i in range(repeats):
            aco.change_init(ant_number=n/3, num_threads=num_threads)
            if num_threads == 0:
                time,cost,solution = aco.do_aco(print_output=False, program="aco_seq")
            else:   
                time,cost,solution = aco.do_aco(print_output=False, program="aco_omp")
            time_list.append(float(time))
        value = (sum(time_list) - min(time_list) - max(time_list)) / (len(time_list) - 2)
        print(f"wątki: {num_threads}\t miasta: {n}\t czas całkowity: {round(sum(time_list), 3)}\t czas średni: {round(sum(time_list) / len(time_list), 3)}\t czas skorygowany: {round(value, 3)}")
        #print(f"wątki: {num_threads}\t miasta: {n}\t czas: {round(sum(time_list) / repeats, 3)}\t czas całkowity: {round(sum(time_list), 3)}")
        row.append(value)
    data.append(row)

df = pd.DataFrame(data, columns=columns)
df.to_csv("results/omp_result.csv")