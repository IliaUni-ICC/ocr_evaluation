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
