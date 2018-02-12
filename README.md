## Optimizers-for-Tensorflow
Adam, NAdam and AAdam optimizers

### Requirements

* Tensorflow (Last version)
* Python (3 or higher)
* Your computer :-)
  
  
### How to use
```
# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = AAdamOptimizer(2e-3).minimize(cross_entropy)
```

### Results 
I included a small [test file](neuralnets-testing-optimizers.py) which implements a simple neural net and 8 optimizers including AAdam and its variant. To run the code, simply go to command line an put ```python neuralnets-testing-optimizers.py```. You don't need tensorflow to run this file and to quickly view the difference between optimizers. 
However, I recommand for more interesting cases that you use the separate pyhton files.

The optimizers are tested on the make_moons toy data set by sklearn, availaible [here](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)

Here are the results so far ... (leraning rate = 1e-2)

```
rmsprop => mean accuracy: 0.8698666666666667, std: 0.007847434117099816
sgd => mean accuracy: 0.8794666666666666, std: 0.001359738536958064
adam => mean accuracy: 0.872, std: 0.009074506414492586
aadam1 => mean accuracy: 0.8741333333333333, std: 0.006607739569794042 <-- without the sign of the gradient
nesterov => mean accuracy: 0.864, std: 0.021496046148071032
aadam2 => mean accuracy: 0.8784000000000001, std: 0.0011313708498984561 <-- with the sign of the gradient
adagrad => mean accuracy: 0.7981333333333334, std: 0.11408036153908735
momentum => mean accuracy: 0.7970666666666667, std: 0.025884529914388783
```

*Enjoy !*

-----
Deep learning = Apprenttissage profond (in french)
Neural nets = Réseaux de neurones (in french)
