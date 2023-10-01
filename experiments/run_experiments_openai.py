from chemlift.icl.utils import LangChainChatModelWrapper
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from chemlift.icl.fewshotclassifier import FewShotClassifier
from chemlift.icl.fewshotpredictor import Strategy
from gptchem.data import get_photoswitch_data
from sklearn.model_selection import train_test_split
from gptchem.evaluator import evaluate_classification
import time
from fastcore.xtras import save_pickle, load_pickle
import os
import dotenv
import langchain
from langchain.cache import SQLiteCache

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
dotenv.load_dotenv("../.env", override=True)

number_support_samples = [5, 10, 20, 50, 100]
strategies = [Strategy.RANDOM, Strategy.DIVERSE]

openai_llm_models = ["text-ada-001", "text-davinci-003"]
openai_chat_models = ["gpt-4", "gpt-3.5-turbo"]

openai_models = openai_llm_models + openai_chat_models


def get_timestr():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def train_test(
    num_support_samples,
    strategy,
    model,
    num_test_points,
    random_state=42,
    temperature=0.8,
    max_test=5,
):
    if model in openai_chat_models:
        llm = LangChainChatModelWrapper(ChatOpenAI(model=model, temperature=temperature))
    elif model in openai_llm_models:
        llm = OpenAI(model_name=model)
    else:
        raise ValueError(f"Unknown model {model}")

    classifier = FewShotClassifier(
        llm,
        property_name="class of the transition wavelength",
        n_support=num_support_samples,
        strategy=strategy,
        seed=random_state,
        prefix="You are an expert chemist. ",
        max_test=max_test,
    )

    data = get_photoswitch_data()
    data = data.dropna(subset=["SMILES", "E isomer pi-pi* wavelength in nm"])

    data["label"] = data["E isomer pi-pi* wavelength in nm"].apply(
        lambda x: 1 if x > data["E isomer pi-pi* wavelength in nm"].median() else 0
    )

    data_train, data_test = train_test_split(
        data, test_size=num_test_points, stratify=data["label"], random_state=random_state
    )

    classifier.fit(data_train["SMILES"].values, data_train["label"].values)
    predictions = classifier.predict(data_test["SMILES"].values)

    report = evaluate_classification(data_test["label"].values, predictions)

    report["num_support_samples"] = num_support_samples
    report["strategy"] = strategy.value
    report["model"] = model
    report["num_test_points"] = num_test_points
    report["random_state"] = random_state

    report["predictions"] = predictions
    report["targets"] = data_test["label"].values
    report["max_test"] = max_test
    report["temperature"] = temperature

    if not os.path.exists("results"):
        os.makedirs("results")

    save_pickle(f"results/{get_timestr()}_openai_report.pkl", report)
    print(report)


if __name__ == "__main__":
    for seed in range(5):
        for num_support_samples in number_support_samples:
            for strategy in strategies:
                for anthropic_mode in openai_models:
                    for num_test_points in [50]:
                        for temperature in [0.2, 0.8]:
                            for max_test in [1, 5, 10]:
                                try:
                                    train_test(
                                        num_support_samples,
                                        strategy,
                                        anthropic_mode,
                                        num_test_points,
                                        random_state=seed + 34,
                                        temperature=temperature,
                                        max_test=max_test,
                                    )
                                except Exception as e:
                                    print(e)
