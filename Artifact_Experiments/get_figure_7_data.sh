cd ../src

# Get the data
python3 size_stat_adapt.py 0 0 > ../Artifact_Experiments/out_adapt_default.txt
python3 size_stat_adapt.py 1 0 > ../Artifact_Experiments/out_adapt_mlonly.txt
python3 size_stat_adapt.py 1 1 > ../Artifact_Experiments/out_adapt_full.txt

python3 size_stat_qcels.py 0 0 > ../Artifact_Experiments/out_qcels_default.txt
python3 size_stat_qcels.py 1 0 > ../Artifact_Experiments/out_qcels_mlonly.txt
python3 size_stat_qcels.py 1 1 > ../Artifact_Experiments/out_qcels_full.txt

# Hard work of parsing:
prefixes=(
  "Seed: 0 qubits: 20hamiltonian"
  "Seed: 1 qubits: 20hamiltonian"
  "Seed: 2 qubits: 20hamiltonian"
  "Seed: 3 qubits: 20hamiltonian"
  "Seed: 4 qubits: 20hamiltonian"
  "Seed: 5 qubits: 20hamiltonian" 
  "Seed: 5 qubits: 24hamiltonian"
  "Seed: 6 qubits: 24hamiltonian"
  "Seed: 7 qubits: 24hamiltonian"
  "Seed: 8 qubits: 24hamiltonian"
  "Seed: 9 qubits: 24hamiltonian"
  "Seed: 0 qubits: 28hamiltonian"
  "Seed: 1 qubits: 28hamiltonian"
  "Seed: 2 qubits: 28hamiltonian"
  "Seed: 3 qubits: 28hamiltonian"
  "Seed: 4 qubits: 28hamiltonian"
  "dummy"
)

echo "  A | B | C | D | E | F | G | H | I"
echo " =================================="

for ((i=0; i<${#prefixes[@]}-1; i++)); do
    rec=`grep -A 1 -B 1 "${prefixes[$i]}" ../Artifact_Experiments/out_adapt_full.txt`
    A=`echo "$rec" | grep "adding ham of size" | cut -d' ' -f6`
    B=`echo "$rec" | grep "(compressed-ML)"  | cut -d' ' -f4 | cut -d')' -f2`
    C=`echo "$rec" | grep "(qp-before)"  | cut -d' ' -f7 | cut -d')' -f2`
    H=`echo "$rec" | grep "round size:"| cut -d' ' -f7` #"ADAPT truncated wt tests"


    rec=`grep -A 1 -B 1 "${prefixes[$i]}" ../Artifact_Experiments/out_qcels_mlonly.txt`
    D=`echo "$rec" | grep "round size:"| cut -d' ' -f7` #"QCELS truncated optimised"

    rec=`grep -A 1 -B 1 "${prefixes[$i]}" ../Artifact_Experiments/out_qcels_default.txt`
    E=`echo "$rec" | grep "round size:"| cut -d' ' -f7` #"QCELS truncated default"

    rec=`grep -A 1 -B 1 "${prefixes[$i]}" ../Artifact_Experiments/out_adapt_mlonly.txt`
    F=`echo "$rec" | grep "round size:"| cut -d' ' -f7` #"ADAPT optimised"

    rec=`grep -A 1 -B 1 "${prefixes[$i]}" ../Artifact_Experiments/out_adapt_default.txt`
    G=`echo "$rec" | grep "round size:"| cut -d' ' -f7` #"ADAPT default"

    rec=`grep -A 1 -B 1 "${prefixes[$i]}" ../Artifact_Experiments/out_qcels_full.txt`
    I=`echo "$rec" | grep "round size:"| cut -d' ' -f7` #"QCELS truncated wt tests"

    echo " $A | $B | $C | $D | $E | $F | $G | $H | $I"
done

echo ">> Mapping:"
echo "   |-A: Size of Hamiltonian"
echo "   |-B: Reduced size of Ham for ML"
echo "   |-C: Ham qp size"
echo "   |-D: QCELS truncated optimised"
echo "   |-E: QCELS truncated default"
echo "   |-F: ADAPT optimised"
echo "   |-G: ADAPT default"
echo "   |-H: ADAPT truncated wt tests"
echo "   |-I: QCELS truncated wt tests"

# Clean up
rm -rf  ../Artifact_Experiments/out_*

echo ">> DONE."
