from chemlift.icl.fewshotclassifier import FewShotClassifier
from chemlift.icl.fewshotpredictor import Strategy
from gptchem.data import get_photoswitch_data
from sklearn.model_selection import train_test_split


def test_fewshotclassifier(get_claude, get_davinci):
    llm = get_claude

    classifier = FewShotClassifier(
        llm,
        property_name="class of the transition wavelength",
        n_support=5,
        strategy=Strategy.RANDOM,
        seed=42,
        prefix="You are an expert chemist. ",
    )

    data = get_photoswitch_data()
    data = data.dropna(subset=["SMILES", "E isomer pi-pi* wavelength in nm"])

    data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
    data_train["label"] = [
        1 if x > data_train["E isomer pi-pi* wavelength in nm"].median() else 0
        for x in data_train["E isomer pi-pi* wavelength in nm"]
    ]
    data_test["label"] = [
        1 if x > data_train["E isomer pi-pi* wavelength in nm"].median() else 0
        for x in data_test["E isomer pi-pi* wavelength in nm"]
    ]

    classifier.fit(data_train["SMILES"].values, data_train["label"].values)

    predictions = classifier.predict(data_test["SMILES"].values[:5])
    assert len(predictions) == 5
    assert isinstance(predictions, list)

    llm = get_davinci

    classifier = FewShotClassifier(
        llm,
        property_name="class of the transition wavelength",
        n_support=5,
        strategy=Strategy.RANDOM,
        seed=42,
        prefix="You are an expert chemist. ",
    )
    classifier.fit(data_train["SMILES"].values, data_train["label"].values)

    predictions = classifier.predict(data_test["SMILES"].values[:5])
    assert len(predictions) == 5
    assert isinstance(predictions, list)
    for pred in predictions:
        assert isinstance(pred, int)
