Getting started
=====================


The first thing you need to do is to decide whether you want to use in-context learning or fine-tuning. 
The choice depends on multiple factors: 

- If you have a lot of training data, they will not fit into the context window of the model.
In this case, the only way to use all data points is to go with fine-tuning
- If you have only a handful of data points, you will be able to achieve good results with finetuning. 
In this case, you might have more luck with in context learning.



Fine-tuning 
...............


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


In-context learning (ICL)
...........................

The is no real "fitting" process in the in-context learning setting.
The only thing that happens if you call :code:`model.fit()` is that we might select the support set. 
In this case, support set refers to the samples that are shown to the model in the prompt. 

For ICL, you need to provide a LangChain LLM model. If you want to use a LangChain chat model, you can use it 
via our wrapper. 

```python
from chemlift.icl.utils import LangChainChatModelWrapper
from chemlift.icl.fewshotclassifier import FewShotClassifier
from langchain.chat_models import ChatAnthropic
from langchain.llms import OpenAI

classifier = FewShotClassifier(LangChainChatModelWrapper(ChatAnthropic()))
# or classifier = FewShotClassifier(OpenAI())
classifier.fit(X, y)
```