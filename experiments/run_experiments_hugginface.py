from langchain import HuggingFaceHub
from chemlift.icl.fewshotclassifier import FewShotClassifier
from chemlift.icl.fewshotpredictor import Strategy
from gptchem.data import get_photoswitch_data
from sklearn.model_selection import train_test_split
from gptchem.evaluator import evaluate_classification
import time
from fastcore.xtras import save_pickle, load_pickle
import os
import dotenv

dotenv.load_dotenv("../.env", override=True)

number_support_samples = [5, 10, 20, 50, 100]
strategies = [Strategy.RANDOM, Strategy.DIVERSE]

models = [
    "google/flan-t5-xl",
    "bigscience/bloom",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
]


context_sizes = {
    "google/flan-t5-xl": 1024,
    "bigscience/bloom": 1024,
    "EleutherAI/pythia-70m-deduped": 500,
    "EleutherAI/pythia-160m-deduped": 500,
    "EleutherAI/pythia-410m-deduped": 500,
    "EleutherAI/pythia-1b-deduped": 500,
    "EleutherAI/pythia-2.8b-deduped": 500,
    "EleutherAI/pythia-6.9b-deduped": 500,
    "EleutherAI/pythia-12b-deduped": 500,
}


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
    llm = HuggingFaceHub(
        repo_id=model, model_kwargs={"temperature": temperature, "max_length": context_sizes[model]}
    )
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

    save_pickle(f"results/{get_timestr()}_huggingface_report.pkl", report)
    print(report)


if __name__ == "__main__":
    for seed in range(5):
        for num_support_samples in number_support_samples:
            for strategy in strategies:
                for anthropic_mode in models:
                    for num_test_points in [50]:
                        for temperature in [0.2, 0.8]:
                            for max_test in [1, 5, 10]:
                                try:
                                    train_test(
                                        num_support_samples,
                                        strategy,
                                        anthropic_mode,
                                        num_test_points,
                                        random_state=seed,
                                        temperature=temperature,
                                        max_test=max_test,
                                    )
                                except Exception as e:
                                    print(anthropic_mode, e)
