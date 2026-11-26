Neuron_dir=../data/neuron_data/multi_neuron_data
Deltas=(2.0 4.0 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0)
for delta in ${Deltas[@]}; do
python 95_multi_neuron_analysis.py --neuron_dir_multi ${Neuron_dir} --delta ${delta}
done
