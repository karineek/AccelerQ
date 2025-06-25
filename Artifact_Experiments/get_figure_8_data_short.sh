cd ../src

## QCELS:
firstSTR=`cat QCELS_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=6,:g" QCELS_answer_experiments.py
for i in {1..2}; do python3 QCELS_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" QCELS_answer_experiments.py
for i in {1..2}; do python3 QCELS_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" QCELS_answer_experiments.py

firstSTR=`cat QCELS_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=7,:g" QCELS_answer_experiments.py
for i in {1..2}; do python3 QCELS_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" QCELS_answer_experiments.py
for i in {1..2}; do python3 QCELS_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" QCELS_answer_experiments.py

firstSTR=`cat first_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=6,:g" first_answer_experiments-tests.py
for i in {1..2}; do python3 first_answer_experiments-tests.py; done

firstSTR=`cat first_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=7,:g" first_answer_experiments-tests.py
for i in {1..2}; do python3 first_answer_experiments-tests.py; done

## ADAPT:
firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=6,:g" first_answer_experiments.py
for i in {1..2}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..2}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=7,:g" first_answer_experiments.py
for i in {1..10}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = True:is_OPT = False:g" first_answer_experiments.py
for i in {1..2}; do python3 first_answer_experiments.py; done
sed -i "s:is_OPT = False:is_OPT = True:g" first_answer_experiments.py

firstSTR=`cat first_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=6,:g" first_answer_experiments-tests.py
for i in {1..2}; do python3 first_answer_experiments-tests.py; done

firstSTR=`cat first_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=7,:g" first_answer_experiments-tests.py
for i in {1..2}; do python3 first_answer_experiments-tests.py; done
