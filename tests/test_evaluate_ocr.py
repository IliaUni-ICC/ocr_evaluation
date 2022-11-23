import copy

from datasets import load_dataset

from src.ocr_evaluation.evaluate.metrics import evaluate_by_words
from src.ocr_evaluation.ocr.fiftyone import FiftyOneOcr
import json


def test_word_evaluation_google_ocr():
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

    annotations_df_ground_truth = FiftyOneOcr(data=dataset_sample).get_word_annotations(convert_bbox=True)
    annotations_df_google_ocr = FiftyOneOcr.from_google_ocr(ocr=gocr).get_word_annotations(convert_bbox=True)
    annotations_df_tesseract_ocr_iliauni = FiftyOneOcr.from_google_ocr(ocr=tocr_iliauni).get_word_annotations(
        convert_bbox=True)
    annotations_df_tesseract_ocr_official = FiftyOneOcr.from_google_ocr(ocr=tocr_official).get_word_annotations(
        convert_bbox=True)

    eval_results_gocr = evaluate_by_words(annotations_df_ground_truth, annotations_df_google_ocr, pref1="Pred_",
                                          pref2="Tar_")

    eval_results_tocr_iliauni = evaluate_by_words(annotations_df_ground_truth, annotations_df_tesseract_ocr_iliauni,
                                                  pref1="Pred_",
                                                  pref2="Tar_")

    eval_results_tocr_official = evaluate_by_words(annotations_df_ground_truth, annotations_df_tesseract_ocr_official,
                                                   pref1="Pred_",
                                                   pref2="Tar_")

    print(eval_results_gocr, eval_results_tocr_iliauni, eval_results_tocr_official)
