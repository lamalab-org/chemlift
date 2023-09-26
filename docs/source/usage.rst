Getting started
=====================

Classification 
-----------------



Regression 
-----------------

Please note that in the LIFT setting regression is not regression in the tradtional sense. 
We do not train the model with a regression loss such as MSE but instead simply continue using the cross-entropy loss and think of predicting (floating point) numbers as next token predictions. 
In this case, the model has no direct feedback that confusing 1 and 2 is worse than confusing 1 and 1000.
Note that we also will not predict numbers with an arbitrary precision but instead round them to a certain number of integers. That is, already this simple rounding will lead to a certain amount of error. 

You can estimate how much error you will get from rounding using the following snippet:

```python

from chemlift.errorestimate import estimate_rounding_error

estimate_rounding_error(y, 2)
```

which will return a dictionary with the best-case regression metrics a perfect model could achieve given this rounding. 
