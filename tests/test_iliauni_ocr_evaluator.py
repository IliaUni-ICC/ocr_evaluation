import copy
import json

from datasets import load_dataset

import evaluate
from src.ocr_evaluation.ocr.fiftyone import FiftyOneOcr


def test_iliauni_ocr_evaluation():
    try:
        import cloudpickle
        with open('mock/sample_1/iliauni_icc_georgian_ocr.pkl', 'rb') as file:
            dataset_sample = cloudpickle.load(file)
    except:
        dataset = load_dataset("anz2/iliauni_icc_georgian_ocr", use_auth_token="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")
        dataset_sample = dataset['test'][0]

    try:
        with open('mock/sample_1/google_ocr.json', 'r') as file:
            gocr = json.load(file)
    except:
        gocr = copy.deepcopy(dataset_sample)

    try:
        with open('mock/sample_1/tesseract_ocr_iliauni.json', 'r') as file:
            tocr_iliauni = json.load(file)
    except:
        tocr_iliauni = copy.deepcopy(dataset_sample)

    try:
        with open('mock/sample_1/tesseract_ocr_official.json', 'r') as file:
            tocr_official = json.load(file)
    except:
        tocr_official = copy.deepcopy(dataset_sample)

    ocr_evaluator = evaluate.load("anz2/iliauniiccocrevaluation")

    ground_truth_data = FiftyOneOcr(data=dataset_sample).data
    google_ocr_data = FiftyOneOcr.from_google_ocr(ocr=gocr).data
    tesseract_ocr_iliauni_data = FiftyOneOcr.from_google_ocr(ocr=tocr_iliauni).data
    tesseract_ocr_official_data = FiftyOneOcr.from_google_ocr(ocr=tocr_official).data

    eval_results = {}
    for eval_method in ["word", "word_group"]:
        eval_result = ocr_evaluator._compute(
            predictions=[google_ocr_data, tesseract_ocr_iliauni_data, tesseract_ocr_official_data],
            references=[ground_truth_data, ground_truth_data, ground_truth_data],
            eval_method=eval_method
        )

        eval_results[eval_method] = eval_result

    assert eval_results
