# This script is for evaluation only. It runs ADAPT-QSCI - Configuration 1 and ADAPT-QSCI - Configuration 2 only, each repeated 10 times.

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=0,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=1,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=2,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=3,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=4,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=5,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=5,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=6,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=7,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=8,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=9,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=0,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=1,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=2,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=3,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=4,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py
