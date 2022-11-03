# https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def bb_intersection_over_union(boxA, boxB):
    EPS = 1e-5
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + EPS) * max(0, yB - yA + EPS)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + EPS) * (boxA[3] - boxA[1] + EPS)
    boxBArea = (boxB[2] - boxB[0] + EPS) * (boxB[3] - boxB[1] + EPS)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def bb_intersection_over_union_vectorized(bboxes1, bboxes2):
    low = np.s_[..., :2]
    high = np.s_[..., 2:]

    EPS = 1e-5

    A, B = bboxes1.copy(), bboxes2.copy()
    A = np.tile(A, (1, len(bboxes2))).reshape(len(bboxes1) * len(bboxes2), -1)
    B = np.tile(B, (len(bboxes1), 1))

    A[high] += EPS
    B[high] += EPS

    intrs = (
        np.maximum(
            0.0,
            np.minimum(
                A[high],
                B[high]
            )
            -
            np.maximum(
                A[low],
                B[low]
            )
        )
    ).prod(-1)

    ious = intrs / ((A[high] - A[low]).prod(-1) + (B[high] - B[low]).prod(-1) - intrs)

    return ious.reshape(len(bboxes1), len(bboxes2))


def bb_is_on_same_line_vectorized(bboxes1, bboxes2):
    low = np.s_[..., 1]
    high = np.s_[..., 3]

    A, B = bboxes1.copy(), bboxes2.copy()
    A = np.tile(A, (1, len(bboxes2))).reshape(len(bboxes1) * len(bboxes2), -1)
    B = np.tile(B, (len(bboxes1), 1))

    is_on_same_line = np.bitwise_and(
        np.bitwise_and(A[low] <= (B[low] + B[high]) / 2, (B[low] + B[high]) / 2 <= A[high]),
        np.bitwise_and(B[low] <= (A[low] + A[high]) / 2, (A[low] + A[high]) / 2 <= B[high]),
    )

    return is_on_same_line.reshape(len(bboxes1), len(bboxes2))


def iou(ocr1, ocr2):
    return bb_intersection_over_union(
        (ocr1['x1'], ocr1['y1'], ocr1['x2'], ocr1['y2']),
        (ocr2['x1'], ocr2['y1'], ocr2['x2'], ocr2['y2'])
    )


def _generate_empty_row(example_row, index):
    """This will generate empty row with empty values but it also generates tiny but valid bounding box
    to avoid exceptions while cropping the image"""

    example_row_dict = example_row.to_dict()
    example_row_dict['page'] = example_row_dict.get('page', 0)
    example_row_dict['block'] = 0
    example_row_dict['paragraph'] = 0
    example_row_dict['word'] = 0
    example_row_dict['x1'] = 0
    example_row_dict['y1'] = 0
    example_row_dict['x2'] = 1
    example_row_dict['y2'] = 1
    example_row_dict['conf'] = 0.0
    example_row_dict['text'] = ""

    empty_row = pd.DataFrame([example_row_dict], columns=example_row.index, index=[index])

    return empty_row


def word_or_symbol_pair_matching(df1, df2, pref1, pref2):
    """Applies IOU based matching of words or symbol elements using rectangular bounding boxes (x1,y1,x2,y2).
    It sorts makes sure that matching between first and second set is unique which means that it's not allowed to have
    one item in two different pairs. If pair isn't found then empty element is used as a pair. This way it's guaranteed
    that word or symbol level matching is correctly evaluated. Pairs are generated in decreasing order of IOU values.
    """
    # match word pairs by page
    text_pairs_dfs_per_page = []
    unique_page_ids = sorted(list(set(df1['page'].unique().tolist() + df2['page'].unique().tolist())))

    for page_id in unique_page_ids:
        # extract words for given page only
        df1_page = df1[df1.page == page_id]
        df2_page = df2[df2.page == page_id]

        if not df1_page.empty and not df1_page.empty:

            # calculate similarities
            similarity_metrics = calculate_ious_fast(ocr1_df=df1_page, ocr2_df=df2_page)
            similarities = []
            for idx1, index1 in enumerate(df1_page.index):
                for idx2, index2 in enumerate(df2_page.index):
                    similarities.append((index1, index2, similarity_metrics[idx1, idx2]))

            # process pair similarities in decreasing order of similarity values
            sorted_similarities = sorted(similarities, key=lambda x: -x[2])
            paired_items_1 = set()
            paired_items_2 = set()
            pairs = []
            for idx1, idx2, similarity in sorted_similarities:
                if idx1 not in paired_items_1 and idx2 not in paired_items_2:
                    if similarity > 0.0:
                        paired_items_1.add(idx1)
                        paired_items_2.add(idx2)
                        pairs.append((idx1, idx2, similarity))

            # add items as empty pairs which weren't matched but index is considered across all pages to avoid collisions
            EMPTY_ITEM_INDEX = max(df1.shape[0], df2.shape[0]) + 100 + page_id
            for idx1, row1 in df1_page.iterrows():
                if idx1 not in paired_items_1:
                    pairs.append((idx1, EMPTY_ITEM_INDEX, 0.0))
            for idx2, row2 in df2_page.iterrows():
                if idx2 not in paired_items_2:
                    pairs.append((EMPTY_ITEM_INDEX, idx2, 0.0))

            # sort pairs according to df2 items original indices
            sorted_pairs = sorted(pairs, key=lambda x: (x[1], x[0]))

            # create row for empty items in each dataframe
            df1_page = pd.concat([df1_page, _generate_empty_row(example_row=df1_page.iloc[0], index=EMPTY_ITEM_INDEX)])
            df2_page = pd.concat([df2_page, _generate_empty_row(example_row=df2_page.iloc[0], index=EMPTY_ITEM_INDEX)])

            # generate pairs dataset
            text_pairs_df = pd.concat(
                [
                    df1_page.loc[[item[0] for item in sorted_pairs], :].reset_index(drop=True).add_prefix(pref1),
                    df2_page.loc[[item[1] for item in sorted_pairs], :].reset_index(drop=True).add_prefix(pref2),
                    pd.DataFrame(
                        data=[item[2] for item in sorted_pairs],
                        columns=["iou"]
                    )
                ],
                axis=1
            )

            text_pairs_dfs_per_page.append(text_pairs_df)

    all_text_pairs_df = pd.concat(text_pairs_dfs_per_page, axis=0)

    return all_text_pairs_df


def word_or_symbol_group_pair_matching(df1, df2, pref1, pref2):
    """Applies IOU based matching of words or symbol elements groups using rectangular bounding boxes (x1,y1,x2,y2).
    It sorts makes sure that matching between first and second set is unique which means that it's not allowed to have
    one item in two different pairs. If pair isn't found then empty element is used as a pair. BUT the difference from
    non-group approach is that here it's possible to match group of words or symbols on each other. This way it's
    more guaranteed that OCR detected result is evaluated correctly.

    Example:
        Let's say we have 2 words: ["abc", "d"] and target has only one word: ["abcd"] then it's better to group first
        two words and match them with the one target word. This way we try to evaluate the overall text detection
        accuracy and not the actual symbol or word boundary detection.

    Note: the grouping operation will happen on one line to avoid unpredictable results if word bounding boxes on
    neighboring lines has some intersection.
    """
    # match word pairs by page
    text_pairs_dfs_per_page = []
    unique_page_ids = sorted(list(set(df1['page'].unique().tolist() + df2['page'].unique().tolist())))

    for page_id in unique_page_ids:
        # extract words for given page only
        df1_page = df1[df1.page == page_id]
        df2_page = df2[df2.page == page_id]

        if not df1_page.empty and not df1_page.empty:
            df1_page_groups, df2_page_groups = get_connected_components(ocr1_df=df1_page, ocr2_df=df2_page)

            # calculate similarities
            similarity_metrics = calculate_ious_fast(ocr1_df=df1_page_groups, ocr2_df=df2_page_groups)
            similarities = []
            for idx1, index1 in enumerate(df1_page_groups.index):
                for idx2, index2 in enumerate(df2_page_groups.index):
                    similarities.append((index1, index2, similarity_metrics[idx1, idx2]))

            # process pair similarities in decreasing order of similarity values
            sorted_similarities = sorted(similarities, key=lambda x: -x[2])
            paired_items_1 = set()
            paired_items_2 = set()
            pairs = []
            for idx1, idx2, similarity in sorted_similarities:
                if idx1 not in paired_items_1 and idx2 not in paired_items_2:
                    if similarity > 0.0:
                        paired_items_1.add(idx1)
                        paired_items_2.add(idx2)
                        pairs.append((idx1, idx2, similarity))

            # add items as empty pairs which weren't matched but index is considered across all pages to avoid collisions
            EMPTY_ITEM_INDEX = max(df1.shape[0], df2.shape[0]) + 100 + page_id
            for idx1, row1 in df1_page_groups.iterrows():
                if idx1 not in paired_items_1:
                    pairs.append((idx1, EMPTY_ITEM_INDEX, 0.0))
            for idx2, row2 in df2_page_groups.iterrows():
                if idx2 not in paired_items_2:
                    pairs.append((EMPTY_ITEM_INDEX, idx2, 0.0))

            # sort pairs according to df2 items original indices
            sorted_pairs = sorted(pairs, key=lambda x: (x[1], x[0]))

            # create row for empty items in each dataframe
            df1_page_groups = pd.concat(
                [df1_page_groups, _generate_empty_row(example_row=df1_page_groups.iloc[0], index=EMPTY_ITEM_INDEX)])
            df2_page_groups = pd.concat(
                [df2_page_groups, _generate_empty_row(example_row=df2_page_groups.iloc[0], index=EMPTY_ITEM_INDEX)])

            # generate pairs dataset
            text_pairs_df = pd.concat(
                [
                    df1_page_groups.loc[[item[0] for item in sorted_pairs], :].reset_index(drop=True).add_prefix(pref1),
                    df2_page_groups.loc[[item[1] for item in sorted_pairs], :].reset_index(drop=True).add_prefix(pref2),
                    pd.DataFrame(
                        data=[item[2] for item in sorted_pairs],
                        columns=["iou"]
                    )
                ],
                axis=1
            )

            text_pairs_dfs_per_page.append(text_pairs_df)

    all_text_pairs_df = pd.concat(text_pairs_dfs_per_page, axis=0)

    return all_text_pairs_df

def calculate_ious_fast(ocr1_df, ocr2_df):
    ious = None
    if not ocr1_df.empty and not ocr2_df.empty:
        bboxes1 = np.array(ocr1_df["bounding_box"].values.tolist())
        bboxes2 = np.array(ocr2_df["bounding_box"].values.tolist())

        if len(bboxes1) > 0 and len(bboxes2) > 0:
            ious = bb_intersection_over_union_vectorized(bboxes1=bboxes1, bboxes2=bboxes2)

    return ious


def calculate_iosl_fast(ocr1_df, ocr2_df):
    iosls = None
    if not ocr1_df.empty and not ocr2_df.empty:
        bboxes1 = np.array(ocr1_df["bounding_box"].values.tolist())
        bboxes2 = np.array(ocr2_df["bounding_box"].values.tolist())

        if len(bboxes1) > 0 and len(bboxes2) > 0:
            iosls = bb_is_on_same_line_vectorized(bboxes1=bboxes1, bboxes2=bboxes2)

    return iosls


def calculate_adjacency_matrix(ocr1_df, ocr2_df):
    """Calculates Adjacency Matrix based on IOU values and for two different sets of items. For each item the adjacency
    is defined by the maximum IOU value. We do 2 sided approach since it can be the case that i is adjacent to j but j
    isn't adjacent to i, so we generate adjacency matrix for directed graph"""
    # concat both dataframes
    ocr_df = pd.concat([ocr1_df, ocr2_df], axis=0).reset_index()

    # calculate ious
    ious = calculate_ious_fast(ocr1_df=ocr_df, ocr2_df=ocr_df)

    # calculate `is on same line` property
    iosls = calculate_iosl_fast(ocr1_df=ocr_df, ocr2_df=ocr_df)

    # build adjacency matrix (1s and 0s)
    adjacency_matrix = np.bitwise_and(ious > 0.0, iosls).astype(np.int)

    return adjacency_matrix


def get_connected_components(ocr1_df, ocr2_df):
    """Apply connected component analysis and group items"""

    def _aggregate_group_items_into_one(df):
        if len(df) == 1:
            return df
        else:
            _df = df.iloc[0, :]
            _bboxes = np.array(df["bounding_box"].values.tolist())


            _df["bounding_box"] = [
                [
                    np.min(_bboxes[:, 0]),
                    np.min(_bboxes[:, 1]),
                    np.max(_bboxes[:, 2]),
                    np.max(_bboxes[:, 3]),
                ]
            ]
            _df["confidence"] = df["confidence"].mean()
            _df["text"] = " ".join(df["text"].tolist())

            return _df

    # 1. calculate adjacency matrix
    adjacency_matrix = calculate_adjacency_matrix(ocr1_df=ocr1_df, ocr2_df=ocr2_df)

    # 2. find connected components
    n_components, labels = connected_components(csgraph=csr_matrix(adjacency_matrix), directed=False,
                                                return_labels=True)

    # 3. separate df1 and df2 items and group for each connected component
    connected_component_groups = pd.Series(labels).to_frame().groupby(0).apply(
        lambda x: {1: [item for item in x.index.tolist() if item < ocr1_df.shape[0]],
                   2: [item - len(ocr1_df) for item in x.index.tolist() if item >= ocr1_df.shape[0]]}).to_dict()

    # 4. check if group of items are consecutive (Optional but interesting)
    # assert np.all(pd.DataFrame(connected_component_groups).loc[1, :].apply(
    #     lambda x: sum(x) == (min(x) * 2 + (len(x) - 1)) * len(x) / 2 if x else True))
    # assert np.all(pd.DataFrame(connected_component_groups).loc[2, :].apply(
    #     lambda x: sum(x) == (min(x) * 2 + (len(x) - 1)) * len(x) / 2 if x else True))

    # 5. merge group items into one
    ocr1_df_groups = pd.concat(
        [
            _aggregate_group_items_into_one(
                ocr1_df.loc[group_data[1], :]
            )
            for group_id, group_data in connected_component_groups.items()
            if group_data[1]
        ],
        axis=0
    ).reset_index(drop=True)

    ocr2_df_groups = pd.concat(
        [
            _aggregate_group_items_into_one(
                ocr2_df.loc[group_data[2], :]
            )
            for group_id, group_data in connected_component_groups.items()
            if group_data[2]
        ],
        axis=0
    ).reset_index(drop=True)

    return ocr1_df_groups, ocr2_df_groups
