This folder contains :
Adam, NAdam and AAdam optimizers

All the 3 versions of the new AAdam work well and outperforms Adam and NAdam in cases that we studied (Logistic Regression, MLP and CNN) on mnist data.

Overall, AAdam01 outperforms AAdam and AAdam02 but further investigations are needed to clearly establish the difference between the three versions. 

### Requirements

* Tensorflow (Last version)
* Python (3 or higher)
  
  
### How to use
First, import the optimizer :
```python
from tf_utils.AAdam import AAdamOptimizer
```
then,
```
# training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = AAdamOptimizer(2e-3).minimize(cross_entropy)
```

