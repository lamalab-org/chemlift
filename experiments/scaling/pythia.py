from gptchem.data import get_photoswitch_data

from chemlift.finetune.peftmodels import PEFTClassifier, ChemLIFTClassifierFactory
from sklearn.model_selection import train_test_split

from fastcore.xtras import load_pickle, save_pickle
from gptchem.evaluator import evaluate_classification
import time
import os


def get_timestr():
    return time.strftime("%Y-%m-%d_%H-%M-%S")


models = [
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-70m-deduped",
]


def train_test(train_size, model_name, random_state=42):
    data = get_photoswitch_data()

    data = data.dropna(subset=["SMILES", "E isomer pi-pi* wavelength in nm"])

    data["binned"] = data["E isomer pi-pi* wavelength in nm"].apply(
        lambda x: 1 if x > data["E isomer pi-pi* wavelength in nm"].median() else 0
    )

    train, test = train_test_split(
        data, train_size=train_size, stratify=data["binned"], random_state=random_state
    )

    train_median = train["E isomer pi-pi* wavelength in nm"].median()
    train["binned"] = train["E isomer pi-pi* wavelength in nm"].apply(
        lambda x: 1 if x > train_median else 0
    )
    test["binned"] = test["E isomer pi-pi* wavelength in nm"].apply(
        lambda x: 1 if x > train_median else 0
    )

    model = ChemLIFTClassifierFactory(
        "transition wavelength class",
        model_name=model_name,
        load_in_8bit=True,
        inference_batch_size=32,
        tokenizer_kwargs={"cutoff_len": 50},
        tune_settings={"num_train_epochs": 32},
    ).create_model()

    model.fit(train["SMILES"].values, train["binned"].values)

    start = time.time()
    predictions = model.predict(test["SMILES"].values)
    end = time.time()

    report = evaluate_classification(test["binned"].values, predictions)

    if not os.path.exists("results"):
        os.makedirs("results")

    outname = f"results/{get_timestr()}_peft_{model_name}_{train_size}.pkl"

    report["model_name"] = model_name
    report["train_size"] = train_size
    report["random_state"] = random_state
    report["predictions"] = predictions
    report["targets"] = test["binned"].values
    report["fine_tune_time"] = model.fine_tune_time
    report["inference_time"] = end - start

    save_pickle(outname, report)


if __name__ == "__main__":
    for seed in range(5):
        for model in models:
            for train_size in [10, 50, 100, 200, 300]:
                train_test(train_size, model, random_state=seed)
