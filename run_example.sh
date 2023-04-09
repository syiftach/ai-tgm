#!/usr/bin/bash
# this is an example usage of local_search.py and result_evaluation.py
printf "***this is an example run of local_search.py***\n"
python3 local_search.py "./example_files/tables_example.csv" "./example_files/reservations_example.csv" -a sa -c=exp
printf "\n***this is an example run of result_evaluation.py***\n"
python3 result_evaluation.py -m "./example_files/solution_input_example.csv" "./example_files/tables_example.csv" -s "solution_reservations_example_sa_exp_T=10_n=10.pkl"