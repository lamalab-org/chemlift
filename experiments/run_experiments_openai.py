from gptchem.data import get_photoswitch_data
from gptchem.evaluator import evaluate_classication

from sklearn.model_selection import train_test_split

openai_models = ["text-ada-001", "text-davinci-003", "gpt-4", "gpt-3.5-turbo"]
