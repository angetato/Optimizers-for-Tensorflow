## Optimizers-for-Tensorflow
Adam, NAdam and AAdam (See below for details about this optimizer) optimizers 

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
*UPDATE:* I included 3 files to test the optimizers on mnist data. I do a small grid search for the learning rate so it can takes some time to execute. I recommand to use a cloud service if you do not have a GPU on your computer (mainly for the MLP and the CNN models). 

I also included a small [test file](neuralnets-testing-optimizers.py) which implements a simple neural net and 8 optimizers including AAdam (the old version) and its variant. To run the code, simply go to command line an put ```python neuralnets-testing-optimizers.py```. You don't need tensorflow to run this file and to quickly view the difference between optimizers. 
However, I recommand for more interesting cases that you use the separated pyhton files.

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

### You want to know about AAdam right ?
The paper about AAdam is under review for a conference. 

I wrote a small [pdf document](AAdam_Summarized.pdf) that summarizes the idea presented in the full paper that I will release on its best version asap.


*Enjoy !*

-----
I'm a native french speaker ... I am currently learning english so if you notice any typos or spelling/grammar errors please let me know, I will be glad to learn from you.
