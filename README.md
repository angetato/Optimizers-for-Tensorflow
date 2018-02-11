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
*Enjoy !*

-----
Deep learning = Apprenttissage profond (in french)
