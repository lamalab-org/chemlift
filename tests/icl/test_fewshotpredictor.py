from chemlift.icl.fewshotpredictor import FewShotPredictor, Strategy
from gptchem.data import get_photoswitch_data
from sklearn.model_selection import train_test_split


def test_fewshotpredictor(get_llm):
    llm = get_llm
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

    predictor = FewShotPredictor(
        llm,
        property_name="class of the transition wavelength",
        n_support=5,
        strategy=Strategy.RANDOM,
        seed=42,
        prefix="You are an expert chemist. ",
    )

    predictor.fit(data_train["SMILES"].values, data_train["label"].values)

    assert predictor._support_set is not None
    x, y = predictor._support_set
    assert len(x) == 5 == len(y)
    formatted = predictor._format_examples(x, y)
    assert isinstance(formatted, str)
    assert formatted.startswith("-")
    parts = formatted.split("\n")
    assert len(parts) == 5
    for part in parts:
        assert "-" in part
        assert ":" in part

    queries = predictor._format_queries(["CC", "CCC"])
    assert isinstance(queries, str)

    prompt = predictor.template.format(
        property_name=predictor._property_name,
        queries=queries,
        examples=formatted,
        number=predictor._n_support,
        materialclass=predictor._materialclass,
        prefix=predictor._prefix,
    )

    assert isinstance(prompt, str)
