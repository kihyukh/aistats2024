# neurips2023

## RWRL experiment
```shell
python .\generate_offline_dataset.py --config .\config\online_alg\rwrl.yaml
python .\run.py --config .\config\PDCA\test.yaml
```

## Tabular experiment
```shell
python .\run_coptidice.py
python .\run_pdca_tabular.py
```