# Efficient Neural Architecture Search via Parameter Sharing

Authors' implementation of "Efficient Neural Architecture Search via Parameter Sharing" (2018) in TensorFlow.

Includes code for CIFAR-10 image classification and Penn Tree Bank language modeling tasks.

Paper: https://arxiv.org/abs/1802.03268

Authors: Hieu Pham*, Melody Y. Guan*, Barret Zoph, Quoc V. Le, Jeff Dean

_This is not an official Google product._

## Penn Treebank

The Penn Treebank dataset is included at `data/ptb`. Depending on the system, you may want to run the script `data/ptb/process.py` to create the `pkl` version. All hyper-parameters are specified in these scripts.

To run the ENAS search process on Penn Treebank, please use the script
```
./scripts/ptb_search.sh
```

To run ENAS with a determined architecture, you have to specify the archiecture using a string. The following is an example script for using the architecture we described in our paper.
```
./scripts/ptb_final.sh
```
A sequence of architecture for a cell with `N` nodes can be specified using a sequence `a` of `2N + 1` tokens

* `a[0]` is a number in `[0, 1, 2, 3]`, specifying the activation function to use at the first cell: `tanh`, `ReLU`, `identity`, and `sigmoid`.
* For each `0 < i < N`, `a[2*i-1]` specifies a previous index and `a[2*i]` specifies the activation function at the `i`-th cell.

For a concrete example, the following sequence specifies the architecture we visualize in our paper

```
0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1
```

<img src="https://github.com/melodyguan/enas/blob/master/img/enas_rnn_cell.png" width="50%"/>

## Notes about parameters

ppl - perplexity, reward function for the treebank RNN

activation functions: 0: tanh, 1: reLU, 2: identity, 3: sigmoid

ptb_final.sh:
```
# what's a logit?
# where do we set M - the number of child models trained duringthe first phase? is that not tunable?
# what's it doing if it already has this architecture?
# Does using fixed_arch and nocontroller_training mean it's just training the final output on Penn Treebank? Does that really take 16h on my CPU?
# note - both Penn Treebank scripts have fixed architectures. CIFAR micro final has fixed, but search does not.
fixed_arc="0 0 0 1 1 2 1 2 0 2 0 5 1 1 0 6 1 8 1 8 1 8 1"
# Section 3.1 'Training Details' has some description of these params.

python src/ptb/main.py \
  --search_for="enas" \
  --reset_output_dir \
  --data_path="data/ptb/ptb.pkl" \
  --output_dir="outputs" \
  --batch_size=64 \
  --child_bptt_steps=35 \ # back-propagation thru time - 35 timesteps mentioned in paper
  --num_epochs=2000 \ # the 2000 epochs mentioned in the paper.
  --child_fixed_arc="${fixed_arc}" \
  --child_rhn_depth=12 \ # rnh?
  --child_num_layers=1 \ # higher for CNNs?
  --child_lstm_hidden_size=748 \
  --child_lstm_e_keep=0.79 \ # some lstm regularization?
  --child_lstm_x_keep=0.25 \
  --child_lstm_h_keep=0.75 \
  --child_lstm_o_keep=0.24 \
  --nochild_lstm_e_skip \ # wut
  --child_grad_bound=0.25 \ # gradient clipping, clip the norm at 0.25 "We find that using a large learning rate whilst clipping the gradient norm at a small threshold makes the updates on ω more stable"
  --child_lr=20.0 \ # decayed by a factor of 0.96 after every epoch starting at epoch 15, for a total of 150 epochs.
  --child_rnn_slowness_reg=1e-3 \ # huh
  --child_l2_reg=5e-7 \ # l2 regularization is wut
  --child_lr_dec_start=14 \ # which iter to start decr?
  --child_lr_dec_every=1 \ # decreasing the learning rate
  --child_lr_dec_rate=0.9991 \ #paper says 0.96
  --child_lr_dec_min=0.001 \ # minimum decrease I guess, says when to stop
  --child_optim_algo="sgd" \ # stochastic gradient descent (vs. ?)
  --log_every=50 \
  --nocontroller_training \ # wut dis
  --controller_selection_threshold=5 \
  --controller_train_every=1 \ # every 1 ... epoch?
  --controller_lr=0.001 \ # Paper says lr=0.00035 
  --controller_sync_replicas \
  --controller_train_steps=100 \
  --controller_num_aggregate=10 \
  --controller_tanh_constant=3.0 \ # Paper says tanh constant = 2.5, "to prevent premature convergence (Bello et al., 2017a;b)"
  --controller_temperature=2.0 \ # Paper says temperature = 5.0, "to prevent premature convergence (Bello et al., 2017a;b),"
  --controller_entropy_weight=0.0001 \ # 'add the controller's sample entropy to the reward, with <this weight>' so they are rewarding the controller for producing high-entropy samples???
  --eval_every_epochs=1 # evalute child models?
  
  ```



## CIFAR-10

To run the experiments on CIFAR-10, please first download the [dataset](https://www.cs.toronto.edu/~kriz/cifar.html). Again, all hyper-parameters are specified in the scripts that we descibe below.

To run the ENAS experiments on the _macro search space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_macro_search.sh
./scripts/cifar10_macro_final.sh
```

A macro architecture for a neural network with `N` layers consists of `N` parts, indexed by `1, 2, 3, ..., N`. Part `i` consists of:

* A number in `[0, 1, 2, 3, 4, 5]` that specifies the operation at layer `i`-th, corresponding to `conv_3x3`, `separable_conv_3x3`, `conv_5x5`, `separable_conv_5x5`, `average_pooling`, `max_pooling`.
* A sequence of `i - 1` numbers, each is either `0` or `1`, indicating whether a skip connection should be formed from a the corresponding past layer to the current layer.

A concrete example can be found in our script `./scripts/cifar10_macro_final.sh`.

To run the ENAS experiments on the _micro search space_ as described in our paper, please use the following scripts:
```
./scripts/cifar10_micro_search.sh
./scripts/cifar10_micro_final.sh
```

A micro cell with `B + 2` blocks can be specified using `B` blocks, corresponding to blocks numbered `2, 3, ..., B+1`, each block consists of `4` numbers
```
index_1, op_1, index_2, op_2
```
Here, `index_1` and `index_2` can be any previous index. `op_1` and `op_2` can be `[0, 1, 2, 3, 4]`, corresponding to `separable_conv_3x3`, `separable_conv_5x5`, `average_pooling`, `max_pooling`, `identity`.

A micro architecture can be specified by two sequences of cells concatenated after each other, as shown in our script `./scripts/cifar10_micro_final.sh`

## Citations

If you happen to use our work, please consider citing our paper.
```
@inproceedings{enas,
  title     = {Efficient Neural Architecture Search via Parameter Sharing},
  author    = {Pham, Hieu and
               Guan, Melody Y. and
               Zoph, Barret and
               Le, Quoc V. and
               Dean, Jeff
  },
  booktitle = {ICML},
  year      = {2018}
}
```


