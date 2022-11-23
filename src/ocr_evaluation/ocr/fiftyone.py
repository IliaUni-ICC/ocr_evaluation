import numpy as np
import pandas as pd


class FiftyOneOcr:
    def __init__(self, data):
        self.data = data

    def get_word_annotations(self, convert_bbox: bool = True) -> pd.DataFrame:
        """Returns dataframe of detections where each row represents independent word annotation

        Args:
            convert_bbox: FiftyOne bounding box type (x1, x2, dx, xy) to 2 point bounding box type (x1, y1, x2, y2)
        """

        annotations = self.data.get("detections", {}).get("detections", {})

        annotations_df = pd.DataFrame(annotations)

        # convert bounding box into 2 point values format
        if convert_bbox:
            bbox = np.array(annotations_df['bounding_box'].values.tolist())
            bbox[:, 2:] += bbox[:, :2]
            annotations_df['bounding_box'] = bbox.tolist()

        return annotations_df

    @classmethod
    def from_google_ocr(cls, ocr: dict):
        data = {
            "detections": {
                "detections": {
                    "bounding_box": [],
                    "confidence": [],
                    "index": [],
                    "page": [],
                    "block": [],
                    "paragraph": [],
                    "word": [],
                    "text": []
                }
            }
        }

        detection_index = 0
        for page_idx, page in enumerate(ocr['pages']):
            for block_idx, block in enumerate(page['blocks']):
                for paragraph_idx, paragraph in enumerate(block['paragraphs']):
                    for word_idx, word in enumerate(paragraph['words']):

                        # extract word text
                        word_text = word.get("text", None)
                        if word_text is None:
                            word_text = ""
                            for symbol in word['symbols']:
                                word_text += symbol['text']

                        if "bounding_box" in word:
                            # extract word bounding box
                            xs = []
                            ys = []
                            for vertex in word['bounding_box']['vertices']:
                                xs.append(vertex['x'])
                                ys.append(vertex['y'])

                            x = min(xs)
                            y = min(ys)
                            width = (max(xs) - x) / float(page['width'])
                            height = (max(ys) - y) / float(page['height'])

                            # normalize x and y later
                            x = x / float(page['width'])
                            y = y / float(page['height'])

                        elif "bbox" in word:
                            x = word['bbox'][0] / float(page['bbox'][2])
                            y = word['bbox'][1] / float(page['bbox'][3])
                            width = (word['bbox'][2] - word['bbox'][0]) / float(page['bbox'][2])
                            height = (word['bbox'][3] - word['bbox'][1]) / float(page['bbox'][3])

                        data["detections"]["detections"]["index"].append(detection_index)
                        data["detections"]["detections"]["bounding_box"].append([x, y, width, height])
                        data["detections"]["detections"]["confidence"].append(word['confidence'])
                        data["detections"]["detections"]["page"].append(page_idx)
                        data["detections"]["detections"]["block"].append(block_idx)
                        data["detections"]["detections"]["paragraph"].append(paragraph_idx)
                        data["detections"]["detections"]["word"].append(word_idx)
                        data["detections"]["detections"]["text"].append(word_text)

                        detection_index += 1

            return FiftyOneOcr(data=data)

    @classmethod
    def from_tesseract_hocr(cls, hocr):
        data = {
            "detections": {
                "detections": {
                    "bounding_box": [],
                    "confidence": [],
                    "index": [],
                    "page": [],
                    "block": [],
                    "paragraph": [],
                    "word": [],
                    "text": []
                }
            }
        }

        detection_index = 0

        """Parses Tesseract OCR response in hocr format converted into BeautifulSoup object"""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(hocr, features="lxml")

        for page_idx, page in enumerate(soup.body.find_all('div', attrs={"class": "ocr_page"})):
            """
            @class:ocr_page
            @id:page_1
            @title:image ""; bbox 0 0 768 1024; ppageno 0
            """
            page_bbox = [int(x) for x in page.attrs["title"].strip().split(";")[1].strip().split(" ")[1:]]
            page_idx = int(page.attrs["title"].strip().split("ppageno")[1].strip())
            page_width = page_bbox[2]
            page_height = page_bbox[2]

            for block_idx, block in enumerate(page.find_all('div', attrs={"class": "ocr_carea"})):
                """
                @class:ocr_carea
                @id:block_1_1
                @title:bbox 89 104 669 196
                """

                for paragraph_idx, paragraph in enumerate(block.find_all('p', attrs={"class": "ocr_par"})):
                    """
                    @class:ocr_par
                    @id:par_1_1
                    @lang:kat
                    @title:bbox 89 104 669 196
                    """

                    for word_idx, word in enumerate(paragraph.find_all('span', attrs={"class": "ocrx_word"})):
                        """
                        @class:ocrx_word
                        @id:word_1_1
                        @title:bbox 90 104 385 141;x_wconf 92
                        """
                        word_bbox = [int(x) for x in word.attrs["title"].split(";")[0].strip().split(" ")[1:]]
                        word_confidence = float(word.attrs['title'].split(";")[1].strip().split(" ")[1]) / 100.0
                        word_symbol_texts = []

                        for symbol_idx, symbol in enumerate(word.find_all('span', attrs={"class": "ocrx_cinfo"})):
                            """
                            @class:ocrx_cinfo
                            @title:x_bboxes 81 26 89 42;x_conf 99.522133
                            """
                            word_symbol_texts.append(symbol.text)

                        word_text = "".join(word_symbol_texts)

                        x = word_bbox[0] / page_width
                        y = word_bbox[1] / page_height
                        width = (word_bbox[2] - word_bbox[0]) / page_width
                        height = (word_bbox[3] - word_bbox[1]) / page_height

                        data["detections"]["detections"]["index"].append(detection_index)
                        data["detections"]["detections"]["bounding_box"].append([x, y, width, height])
                        data["detections"]["detections"]["confidence"].append(word_confidence)
                        data["detections"]["detections"]["page"].append(page_idx)
                        data["detections"]["detections"]["block"].append(block_idx)
                        data["detections"]["detections"]["paragraph"].append(paragraph_idx)
                        data["detections"]["detections"]["word"].append(word_idx)
                        data["detections"]["detections"]["text"].append(word_text)

                        detection_index += 1

            return FiftyOneOcr(data=data)
