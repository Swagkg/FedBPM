##FedBPM: A Decentralized Federated Meta-Method for Heterogeneous and Complex Image Classification via Multi-Scale Feature Fusion
### For FEMNIST

FedBPM:
```bash
python run.py --algo=fedbpm \
--eval_on_test_every=1 \
--dataset=femnist_p_0.2 \
--lr=0.001 \
--num_epochs=1 \
--model=Patt \
--clients_per_round=4 \
--batch_size=10 \
--data_format=pkl \
--num_rounds=2000 \
--meta_algo=maml \
--outer_lr=0.001 \
--result_prefix=./fedbpm_result \
--device=cuda:0 \
--save_every=1000 \
--meta_inner_step=5
``` 

FedAvg:

```bash
--algo=fedavg_adv \
--eval_on_test_every=1 \
--dataset=femnist_p_0.2 \
--lr=1e-4 \
--num_epochs=1 \
--model=cnn \
--clients_per_round=4 \
--batch_size=10 \
--data_format=pkl \
--num_rounds=2000 \
--result_prefix=./fedbpm_result \
--device=cuda:0 \
--save_every=1000 
```