firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=0,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=1,:g" QCELS_anser_experiments.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=2,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=3,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=4,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=20,seed=5,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=5,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=6,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=7,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=8,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=24,seed=9,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=0,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=1,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=2,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=3,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done

firstSTR=`cat QCELS_answer_experiments-tests.py | grep "print(run_algorithm.get_result(n_qubits=" | cut -d'(' -f3 | cut -d' ' -f1`
sed -i "s:$firstSTR:n_qubits=28,seed=4,:g" QCELS_answer_experiments-tests.py
for i in {1..10}; do python3 QCELS_answer_experiments-tests.py; done
