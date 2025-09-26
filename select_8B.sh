export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=False

type=NIRVANA
# type=Pruner

data=bookcorpus


batch_size=8



seq_len=128

sparsity=0.5

iterative_steps=$1

args=(
--base_model meta-llama/Llama-3.1-8B
--device cuda


# Prune setting
--prune_type $type
--prune
--structure_prune
--train

# Calibration setting
--data $data
--num_examples $num_examples
--seq_len $seq_len
--batch_size $batch_size
--max_seq_len 2048
--sparsity $sparsity
--seed 0
--iterative_steps $iterative_steps
--lr 1e-2

--train_samples 4096

# Model setting
--gamma 3.36 # Default gamma value
--test_after_prune
--save_model /PUT/YOUR/PATH/HERE
)


python data_select.py "${args[@]}"