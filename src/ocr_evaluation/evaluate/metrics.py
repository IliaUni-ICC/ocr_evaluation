import pandas as pd

from ocr_evaluation.evaluate.iou import word_or_symbol_pair_matching, word_or_symbol_group_pair_matching


def text_accuracy(df, pref_1, pref_2):
    return (df[f'{pref_1}text'] == df[f'{pref_2}text']).sum() / df.shape[0]


def text_precision(df, pref_1, pref_2):
    ocr1_nonempty = df[f'{pref_1}text'].apply(lambda x: bool(x))
    ocr1 = df[f'{pref_1}text']
    ocr2 = df[f'{pref_2}text']
    return (ocr1_nonempty & (ocr1 == ocr2)).sum() / ocr1_nonempty.sum()


def text_recall(df, pref_1, pref_2):
    ocr2_nonempty = df[f'{pref_2}text'].apply(lambda x: bool(x))
    ocr2 = df[f'{pref_1}text']
    ocr1 = df[f'{pref_2}text']
    return (ocr2_nonempty & (ocr2 == ocr1)).sum() / ocr2_nonempty.sum()


def text_f1(df, pref_1, pref_2):
    precision = text_precision(df, pref_1, pref_2)
    recall = text_recall(df, pref_1, pref_2)

    if precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return f1


def symbol_confusion_matrix(df, pref_1, pref_2):
    all_symbols = list(sorted(set(df[f'{pref_1}text'].tolist() + df[f'{pref_2}text'].tolist())))
    pair_value_counts = df[
        [f'{pref_1}text', f'{pref_2}text']
    ].value_counts()

    pair_cnts = pair_value_counts.reset_index().rename({0: "count"}, axis=1).sort_values(
        by=[f'{pref_1}text', f'{pref_2}text'], ascending=True)

    pair_value_counts_dict = pair_value_counts.to_dict()

    confusion_matrix = pd.DataFrame(
        [
            [pair_value_counts_dict.get((symbol1, symbol2), 0) for symbol2 in all_symbols]
            for symbol1 in all_symbols
        ],
        columns=all_symbols,
        index=all_symbols,
    )

    return confusion_matrix, pair_cnts


def levenstein(text1, text2):
    """Measures the metrics based on edit operations.
    - levenstein_distance: number of character operations (insertion, deletion, substitution) that
        required to get text2 from text1
    - levenstein_similarity: number of matches divided by the number of all operations (fraction of characters that
        don't require modification while transforming text1 into text2)
    - edit_operations: list of character operations (<operation name>, <text1 character>, <text2 character>)
    """
    levenstein_distance, edit_operations = edit_distance(text1, text2)
    if levenstein_distance == 0:
        levenstein_similarity = 1.0
    else:
        matches_cnt = len([item for item in edit_operations if item[0] == "match"])
        all_operations_cnt = len(edit_operations)

        if matches_cnt == 0:
            levenstein_similarity = 0.0
        else:
            levenstein_similarity = float(matches_cnt / all_operations_cnt)

    return levenstein_similarity, levenstein_distance, edit_operations


def edit_distance(text1, text2):
    """
    we have three allowed edit operations:
    - Insert a character
    - Delete a character
    - Substitute a character
    Each of these operations has cost of 1
    Our goal is to minimize number of required operations to convert text1 into text2
    This DP problem which is being solved with 2d array (NxM) where N is the length of text1 and M - length of
    text2.

    DP[i][j]: this is minimum amount of operations to convert text1[:i] into text2[:j]
    The update rule is the following:
    DP[i][j] = min of the following

    case 1: DP[i-1][j-1] # match
    case 2: DP[i-1][j] + 1 # insertion,
    case 3: DP[i][j-1] + 1 # deletion
    case 4: DP[i-1][j-1] + 1 # substitution

    Example:
    text1 = "horse"
    text2 = "ros"

    DP _  r  o  s
    _ [0, 1, 2, 3]
    h [1, 1, 2, 3]
    o [2, 2, 1, 2]
    r [3, 2, 2, 2]
    s [4, 3, 3, 2]
    e [5, 4, 4, 3]
    """
    if not text1:
        return len(text2), []
    elif not text2:
        return len(text1), []

    INF = 10 ** 10
    N = len(text1)
    M = len(text2)

    DP = [[INF for _ in range(M + 1)] for _ in range(N + 1)]
    P = [[None for _ in range(M + 1)] for _ in range(N + 1)]

    for i in range(N + 1):
        DP[i][0] = i
        P[i][0] = "insertion"
    for j in range(M + 1):
        DP[0][j] = j
        P[0][j] = "deletion"

    for j in range(1, M + 1):
        for i in range(1, N + 1):

            pair_mismatch = int(text1[i - 1] != text2[j - 1])
            match_case = None
            match_cost = INF

            # match
            if match_cost > DP[i - 1][j - 1] + pair_mismatch:
                match_cost = DP[i - 1][j - 1] + pair_mismatch
                match_case = "substitution" if pair_mismatch == 1 else "match"

            # insertion
            if match_cost > DP[i - 1][j] + 1:
                match_cost = DP[i - 1][j] + 1
                match_case = "insertion"

            # deletion
            if match_cost > DP[i][j - 1] + 1:
                match_cost = DP[i][j - 1] + 1
                match_case = "deletion"

            DP[i][j] = match_cost
            P[i][j] = match_case

    operations = []
    i = N
    j = M
    while (i >= 0 and j >= 0) and not (i == 0 and j == 0):
        if P[i][j] == "substitution":
            operations.append(("substitution", text1[i - 1] if i - 1 >= 0 else "",
                               text2[j - 1] if j - 1 >= 0 else "", i - 1, j - 1))
            i -= 1
            j -= 1
        elif P[i][j] == "match":
            operations.append(
                ("match", text1[i - 1] if i - 1 >= 0 else "", text2[j - 1] if j - 1 >= 0 else "", i - 1, j - 1))
            i -= 1
            j -= 1
        elif P[i][j] == "insertion":
            operations.append(("insertion", text1[i - 1] if i - 1 >= 0 else "",
                               "", i - 1, j - 1))
            i -= 1
        elif P[i][j] == "deletion":
            operations.append(("deletion", "",
                               text2[j - 1] if j - 1 >= 0 else "", i - 1, j - 1))
            j -= 1

    levenstein_distance = DP[N][M]
    operations = operations[::-1]

    return levenstein_distance, operations


def levenstein_metrics(df, pref_1="Pred_", pref_2='Tar_'):
    levenstein_results = df[[f'{pref_1}text', f'{pref_2}text']].apply(
        lambda x: levenstein(text1=x[f'{pref_1}text'], text2=x[f'{pref_2}text']),
        axis=1
    )
    levenstein_similarities = levenstein_results.apply(lambda x: x[0])
    levenstein_distances = levenstein_results.apply(lambda x: x[1])
    edit_operations = levenstein_results.apply(lambda x: x[2])

    return levenstein_similarities, levenstein_distances, edit_operations


def evaluate_by_words(pred_df, target_df, pred_pref='Pred_', target_pref='Target_', **kwargs):
    if not pred_df.empty and not target_df.empty:

        show_hist = kwargs.get("show_hist", False)

        word_pairs_df = word_or_symbol_pair_matching(df1=pred_df, df2=target_df, pref1=pred_pref, pref2=target_pref)

        _similarities, _distances, _edits = levenstein_metrics(df=word_pairs_df, pref_1=pred_pref, pref_2=target_pref)

        word_pairs_df['levenstein_similarities'] = _similarities
        word_pairs_df['levenstein_distances'] = _distances
        word_pairs_df['edit_operations'] = _edits

        edit_operations_stats = {
            operation_id: pd.Series(
                word_pairs_df['edit_operations'].apply(
                    lambda x: [f"[{item[1]}]_[{item[2]}]" for item in x if item[0] == operation_id]
                ).sum(axis=0)).value_counts().to_dict()
            for operation_id in ["insertion", "deletion", "substitution"]
        }

        report = {
            "accuracy": text_accuracy(df=word_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "precision": text_precision(df=word_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "recall": text_recall(df=word_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "f1": text_f1(df=word_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "levenstein_similarities_stats": {
                "mean": word_pairs_df['levenstein_similarities'].mean(),
                "std": word_pairs_df['levenstein_similarities'].std()
            },
            "levenstein_distances_stats": {
                "mean": word_pairs_df['levenstein_distances'].mean(),
                "std": word_pairs_df['levenstein_distances'].std()
            },
            "edit_operations_stats": edit_operations_stats,
            "iou_stats": {
                "mean": word_pairs_df['iou'].mean(),
                "std": word_pairs_df['iou'].std()
            },
            "word_pairs_dataframe": word_pairs_df
        }

        if show_hist is True:
            pd.Series(word_pairs_df['levenstein_similarities']).plot(kind='hist', bins=20,
                                                                     title="Levestein Similarities")
            pd.Series(word_pairs_df['levenstein_distances']).plot(kind='hist', bins=20, title="Levestein Distances")
            for edit_operation_id, edit_operation_data in edit_operations_stats.items():
                pd.Series(edit_operation_data).plot(kind='barh', title=f"{edit_operation_id.capitalize()} Stats")

    else:
        report = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "levenstein_distances_stats": {},
            "levenstein_similarities_stats": {},
            "edit_operations_stats": {key: {} for key in ["insertion", "deletion", "substitution"]},
            "iou_stats": {},
            "word_pairs_dataframe": pd.DataFrame()
        }

    return report


def evaluate_by_word_groups(pred_df, target_df, pred_pref='Pred_', target_pref='Target_', **kwargs):
    if not pred_df.empty and not target_df.empty:

        show_hist = kwargs.get("show_hist", False)

        word_group_pairs_df = word_or_symbol_group_pair_matching(df1=pred_df, df2=target_df, pref1=pred_pref,
                                                                 pref2=target_pref)

        _similarities, _distances, _edits = levenstein_metrics(df=word_group_pairs_df, pref_1=pred_pref,
                                                               pref_2=target_pref)

        word_group_pairs_df['levenstein_similarities'] = _similarities
        word_group_pairs_df['levenstein_distances'] = _distances
        word_group_pairs_df['edit_operations'] = _edits

        edit_operations_stats = {
            operation_id: pd.Series(
                word_group_pairs_df['edit_operations'].apply(
                    lambda x: [f"[{item[1]}]_[{item[2]}]" for item in x if item[0] == operation_id]
                ).sum(axis=0)).value_counts().to_dict()
            for operation_id in ["insertion", "deletion", "substitution"]
        }

        report = {
            "accuracy": text_accuracy(df=word_group_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "precision": text_precision(df=word_group_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "recall": text_recall(df=word_group_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "f1": text_f1(df=word_group_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "levenstein_similarities_stats": {
                "mean": word_group_pairs_df['levenstein_similarities'].mean(),
                "std": word_group_pairs_df['levenstein_similarities'].std()
            },
            "levenstein_distances_stats": {
                "mean": word_group_pairs_df['levenstein_distances'].mean(),
                "std": word_group_pairs_df['levenstein_distances'].std()
            },
            "edit_operations_stats": edit_operations_stats,
            "iou_stats": {
                "mean": word_group_pairs_df['iou'].mean(),
                "std": word_group_pairs_df['iou'].std()
            },
            "word_group_pairs_dataframe": word_group_pairs_df
        }

        if show_hist is True:
            pd.Series(word_group_pairs_df['levenstein_similarities']).plot(kind='hist', bins=20,
                                                                           title="Levestein Similarities")
            pd.Series(word_group_pairs_df['levenstein_distances']).plot(kind='hist', bins=20,
                                                                        title="Levestein Distances")
            for edit_operation_id, edit_operation_data in edit_operations_stats.items():
                pd.Series(edit_operation_data).plot(kind='barh', title=f"{edit_operation_id.capitalize()} Stats")

    else:
        report = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "levenstein_distances_stats": {},
            "levenstein_similarities_stats": {},
            "edit_operations_stats": {key: {} for key in ["insertion", "deletion", "substitution"]},
            "iou_stats": {},
            "word_group_pairs_dataframe": pd.DataFrame()
        }

    return report


def evaluate_by_symbols(pred_df, target_df, pred_pref='Pred_', target_pref='Target_', **kwargs):
    if not pred_df.empty and not target_df.empty:

        show_hist = kwargs.get("show_hist", False)
        symbol_pairs_df = word_or_symbol_pair_matching(df1=pred_df, df2=target_df, pref1=pred_pref, pref2=target_pref)

        confusion_matrix, pair_counts = symbol_confusion_matrix(symbol_pairs_df, pref_1=pred_pref, pref_2=target_pref)

        if show_hist is True:
            pd.Series(pair_counts).plot(kind='barh', title="Symbol Pair Counts")

        report = {
            "accuracy": text_accuracy(df=symbol_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "precision": text_precision(df=symbol_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "recall": text_recall(df=symbol_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "f1": text_f1(df=symbol_pairs_df, pref_1=pred_pref, pref_2=target_pref),
            "confusion_matrix": confusion_matrix,
            "pair_counts": pair_counts,
            "iou_stats": {
                "mean": symbol_pairs_df['iou'].mean(),
                "std": symbol_pairs_df['iou'].std()
            },
        }
    else:
        report = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "confusion_matrix": pd.DataFrame(),
            "pair_counts": pd.DataFrame(),
            "iou_stats": {},
        }

    return report
