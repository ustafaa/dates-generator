# Dates Generator -- Submission

## How to run inference

```
conda env create -f environment.yml
conda activate dates-generator
cd model
python predict.py -i ../data/example_input.txt -o ../predictions.txt
```

Default model: `cvae` (highest val CSR_all of the four).

To use a different model: `--model cvae | cgan | ar | diffusion`.

## Final results

(Run on val set; see report for full breakdown.)

| Model         | val CSR_all | test_random CSR_all | held_out_tuples CSR_all |
|---------------|-------------|----------------------|--------------------------|
| random        | 0.093       | 0.092                | 0.163                    |
| smart_random  | 1.000       | 1.000                | 1.000                    |
| cvae          | 0.978       | 0.979                | 1.000                    |
| cgan          | 0.105       | 0.107                | 0.170                    |
| ar            | 0.140       | 0.144                | 0.149                    |
| diffusion     | 0.134       | 0.138                | 0.113                    |
