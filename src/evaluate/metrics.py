import pandas as pd

from evaluation.iou import word_or_symbol_pair_matching, word_or_symbol_group_pair_matching


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
        text_pairs = word_or_symbol_pair_matching(df1=pred_df, df2=target_df, pref1=pred_pref, pref2=target_pref)
        levenstein_similarities, levenstein_distances, edit_operations = levenstein_metrics(
            df=text_pairs, pref_1=pred_pref, pref_2=target_pref
        )

        levenstein_similarities_stats = {
            **levenstein_similarities.describe().to_dict(),
            "values": levenstein_similarities.tolist()
        }
        levenstein_distances_stats = {
            **levenstein_distances.describe().to_dict(),
            "values": levenstein_distances.tolist()
        }
        iou_stats = {
            **text_pairs.iou.describe().to_dict(),
            "values": text_pairs.iou.tolist()
        }
        edit_operations_stats = {
            operation_id: pd.Series(
                edit_operations.apply(
                    lambda x: [f"[{item[1]}]_[{item[2]}]" for item in x if item[0] == operation_id]
                ).sum(axis=0)).value_counts().to_dict()
            for operation_id in ["insertion", "deletion", "substitution"]
        }

        if show_hist is True:
            pd.Series(levenstein_similarities).plot(kind='hist', bins=20, title="Levestein Similarities")
            pd.Series(levenstein_distances).plot(kind='hist', bins=20, title="Levestein Distances")
            for edit_operation_id, edit_operation_data in edit_operations_stats.items():
                pd.Series(edit_operation_data).plot(kind='barh', title=f"{edit_operation_id.capitalize()} Stats")

        report = {
            "accuracy": text_accuracy(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "precision": text_precision(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "recall": text_recall(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "f1": text_f1(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "levenstein_distances_stats": levenstein_distances_stats,
            "levenstein_similarities_stats": levenstein_similarities_stats,
            "iou_stats": iou_stats,
            "edit_operations_stats": edit_operations_stats,
        }
    else:
        report = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "levenstein_distances_stats": {},
            "levenstein_similarities_stats": {},
            "iou_stats": {},
            "edit_operations_stats": {key: {} for key in ["insertion", "deletion", "substitution"]},
        }

    return report


def evaluate_by_word_groups(pred_df, target_df, pred_pref='Pred_', target_pref='Target_', **kwargs):
    if not pred_df.empty and not target_df.empty:

        show_hist = kwargs.get("show_hist", False)
        text_pairs = word_or_symbol_group_pair_matching(df1=pred_df, df2=target_df, pref1=pred_pref, pref2=target_pref)
        levenstein_similarities, levenstein_distances, edit_operations = levenstein_metrics(
            df=text_pairs, pref_1=pred_pref, pref_2=target_pref
        )

        levenstein_similarities_stats = {
            **levenstein_similarities.describe().to_dict(),
            "values": levenstein_similarities.tolist()
        }
        levenstein_distances_stats = {
            **levenstein_distances.describe().to_dict(),
            "values": levenstein_distances.tolist()
        }
        iou_stats = {
            **text_pairs.iou.describe().to_dict(),
            "values": text_pairs.iou.tolist()
        }
        edit_operations_stats = {
            operation_id: pd.Series(
                edit_operations.apply(
                    lambda x: [f"[{item[1]}]_[{item[2]}]" for item in x if item[0] == operation_id]
                ).sum(axis=0)).value_counts().to_dict()
            for operation_id in ["insertion", "deletion", "substitution"]
        }

        if show_hist is True:
            pd.Series(levenstein_similarities).plot(kind='hist', bins=20, title="Levestein Similarities")
            pd.Series(levenstein_distances).plot(kind='hist', bins=20, title="Levestein Distances")
            for edit_operation_id, edit_operation_data in edit_operations_stats.items():
                pd.Series(edit_operation_data).plot(kind='barh', title=f"{edit_operation_id.capitalize()} Stats")

        report = {
            "accuracy": text_accuracy(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "precision": text_precision(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "recall": text_recall(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "f1": text_f1(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "levenstein_distances_stats": levenstein_distances_stats,
            "levenstein_similarities_stats": levenstein_similarities_stats,
            "iou_stats": iou_stats,
            "edit_operations_stats": edit_operations_stats,
        }
    else:
        report = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "levenstein_distances_stats": {},
            "levenstein_similarities_stats": {},
            "iou_stats": {},
            "edit_operations_stats": {key: {} for key in ["insertion", "deletion", "substitution"]},
        }

    return report


def reduce_word_evaluation_results(eval_results):
    if eval_results:
        accuracies = pd.Series([item['accuracy'] for item in eval_results])
        precisions = pd.Series([item['precision'] for item in eval_results])
        recalls = pd.Series([item['recall'] for item in eval_results])
        f1s = pd.Series([item['f1'] for item in eval_results])
        levenstein_similarities = pd.Series(
            [
                pd.Series(item['levenstein_similarities_stats'].get('values', [])).mean()
                for item in eval_results
            ]
        )
        levenstein_distances = pd.Series(
            [
                pd.Series(item['levenstein_distances_stats'].get('values', [])).mean()
                for item in eval_results
            ]
        )
        ious = pd.Series(
            [
                pd.Series(item['iou_stats'].get('values', [])).mean()
                for item in eval_results
            ]
        )

        levenstein_similarities_stats = {
            **levenstein_similarities.describe().to_dict(),
            "values": levenstein_similarities.tolist()
        }
        levenstein_distances_stats = {
            **levenstein_distances.describe().to_dict(),
            "values": levenstein_distances.tolist()
        }
        iou_stats = {
            **ious.describe().to_dict(),
            "values": ious.tolist()
        }

        edit_operations_stats = {}
        for eval_result in eval_results:
            for edit_operation, edit_operation_data in eval_result['edit_operations_stats'].items():
                if edit_operation not in edit_operations_stats:
                    edit_operations_stats[edit_operation] = {}

                for key, count in edit_operation_data.items():
                    edit_operations_stats[edit_operation][key] = edit_operations_stats[edit_operation].get(key,
                                                                                                           0) + count

        summary = {
            "accuracy": {
                "mean": accuracies.mean(),
                "std": accuracies.std(),
                "values": accuracies.tolist()
            },
            "precision": {
                "mean": precisions.mean(),
                "std": precisions.std(),
                "values": precisions.tolist(),
            },
            "recall": {
                "mean": recalls.mean(),
                "std": recalls.std(),
                "values": recalls.tolist(),
            },
            "f1": {
                "mean": f1s.mean(),
                "std": f1s.std(),
                "values": f1s.tolist(),
            },
            "document_count": len(eval_results),
            "levenstein_distances_stats": levenstein_distances_stats,
            "levenstein_similarities_stats": levenstein_similarities_stats,
            "iou_stats": iou_stats,
            "edit_operations_stats": edit_operations_stats,
        }


    else:
        summary = {
            "accuracy": {},
            "precision": {},
            "recall": {},
            "f1": {},
            "document_count": 0,
            "levenstein_distances_stats": {},
            "levenstein_similarities_stats": {},
            "iou_stats": {},
            "edit_operations_stats": {key: {} for key in ["insertion", "deletion", "substitution"]},
        }

    return summary


def evaluate_by_symbols(pred_df, target_df, pred_pref='Pred_', target_pref='Target_', **kwargs):
    if not pred_df.empty and not target_df.empty:

        show_hist = kwargs.get("show_hist", False)
        text_pairs = word_or_symbol_pair_matching(df1=pred_df, df2=target_df, pref1=pred_pref, pref2=target_pref)

        confusion_matrix, pair_counts = symbol_confusion_matrix(text_pairs, pref_1=pred_pref, pref_2=target_pref)

        iou_stats = {
            **text_pairs.iou.describe().to_dict(),
            "values": text_pairs.iou.tolist()
        }

        if show_hist is True:
            pd.Series(pair_counts).plot(kind='barh', title="Symbol Pair Counts")

        report = {
            "accuracy": text_accuracy(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "precision": text_precision(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "recall": text_recall(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "f1": text_f1(df=text_pairs, pref_1=pred_pref, pref_2=target_pref),
            "confusion_matrix": confusion_matrix,
            "pair_counts": pair_counts,
            "iou_stats": iou_stats,
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


def reduce_pair_counts(pair_counts):
    reduced_pair_counts_df = pd.DataFrame()
    columns = []
    if pair_counts:
        pair_counts_dict = {}
        for pair_count in pair_counts:
            if not pair_count.empty:
                pair_count_dict = pair_count.set_index(pair_count.columns[:-1].tolist(), drop=True).to_dict()[
                    pair_count.columns[-1]]
                columns = pair_count.columns.tolist()
            else:
                pair_count_dict = {}

            for key, value in pair_count_dict.items():
                pair_counts_dict[key] = pair_counts_dict.get(key, 0) + value

        reduced_pair_counts_df = pd.Series(pair_counts_dict).to_frame().reset_index()
        if columns:
            reduced_pair_counts_df.columns = columns

    return reduced_pair_counts_df


def reduce_confusion_matrices(confusion_matrices):
    reduced_confusion_matrices_df = pd.DataFrame()
    if confusion_matrices:
        all_index_values = set()
        confusion_matrices_dict = {}
        for confusion_matrix in confusion_matrices:
            if not confusion_matrix.empty:
                confusion_matrix_dict = {
                    (index, column): confusion_matrix.loc[index, column]
                    for index in confusion_matrix.index
                    for column in confusion_matrix.columns
                }
            else:
                confusion_matrix_dict = {}

            for key, value in confusion_matrix_dict.items():
                all_index_values.add(key[0])
                all_index_values.add(key[1])
                confusion_matrices_dict[key] = confusion_matrices_dict.get(key, 0) + value

        all_index_values = list(sorted(list(all_index_values)))
        reduced_confusion_matrices_df = pd.DataFrame(
            [
                [
                    confusion_matrices_dict.get((index, column), 0)
                    for column in all_index_values
                ]
                for index in all_index_values
            ],
            columns=all_index_values,
            index=all_index_values,
        )

    return reduced_confusion_matrices_df


def reduce_symbol_evaluation_results(eval_results):
    """
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
    """
    if eval_results:
        accuracies = pd.Series([item['accuracy'] for item in eval_results])
        precisions = pd.Series([item['precision'] for item in eval_results])
        recalls = pd.Series([item['recall'] for item in eval_results])
        f1s = pd.Series([item['f1'] for item in eval_results])
        confusion_matrices = [item['confusion_matrix'] for item in eval_results]
        pair_counts = [item['pair_counts'] for item in eval_results]
        ious = pd.Series(
            [
                pd.Series(item['iou_stats'].get('values', [])).mean()
                for item in eval_results
            ]
        )

        iou_stats = {
            **ious.describe().to_dict(),
            "values": ious.tolist()
        }

        summary = {
            "accuracy": {
                "mean": accuracies.mean(),
                "std": accuracies.std(),
                "values": accuracies.tolist()
            },
            "precision": {
                "mean": precisions.mean(),
                "std": precisions.std(),
                "values": precisions.tolist(),
            },
            "recall": {
                "mean": recalls.mean(),
                "std": recalls.std(),
                "values": recalls.tolist(),
            },
            "f1": {
                "mean": f1s.mean(),
                "std": f1s.std(),
                "values": f1s.tolist(),
            },
            "document_count": len(eval_results),
            "pair_counts": reduce_pair_counts(pair_counts),
            "confusion_matrix": reduce_confusion_matrices(confusion_matrices),
            "iou_stats": iou_stats,
        }

    else:
        summary = {
            "accuracy": {},
            "precision": {},
            "recall": {},
            "f1": {},
            "document_count": 0,
            "pair_counts": pd.DataFrame(),
            "confusion_matrix": pd.DataFrame(),
            "iou_stats": {},
        }

    return summary
