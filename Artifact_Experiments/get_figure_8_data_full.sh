cd ../src

python3 run-exp-sim-qcels-tests.sh > sim_res_qcels-tests.log 2>&1 
python3 run-exp-sim-qcels.sh > sim_res_qcels.log 2>&1 
python3 run-exp-sim-tests.sh > sim_res_adapt-tests.log 2>&1 
python3 run-exp-sim.sh > sim_res_adapt-tests.log 2>&1 
