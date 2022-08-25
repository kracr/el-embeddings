# bash script to evaluating trained model


for dim in 100
do
  for margin in 0
  do
    for type in 1_1 1_n n_n
    do
      python Evaluating_HITS.py --data SNOMED --tag $type --margin $margin --dim $dim --early_stop True > test_results/SNOMED/inference_$type\_{$dim}_{$margin}_{1000}.txt
      # python Evaluating_HITS.py --tag $type --margin $margin --dim $dim > test_results_no_early_stopping/result_$type\_{$dim}_{$margin}_{1000}.txt
      # echo result_$type\_{$dim}_{$margin}_{1000}.txt
    done
  done 
done

# python Evaluating_HITS.py --tag 1_n > result_1_n_{100}_{-0.1}_{1000}.txt 

# python Evaluating_HITS.py --tag n_n > result_n_n_{100}_{-0.1}_{1000}.txt 