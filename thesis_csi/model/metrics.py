try:
    import evaluate
except ImportError:
    pass
import numpy as np

from thesis_csi.logging import get_logger

logger = get_logger(__name__)


def average_precision(relevant_documents: np.ndarray, distances: np.ndarray, epsilon: float = 1e-12) -> float:
    """Compute average precision. If no threshold is given, the predictions and true values are
       assumed to be sorted.

    Args:
        relevant_documents: An array of shape (n_documents,) containing the relevant documents.
        distances: An array of shape (n_documents,) containing the distances for each document.
        epsilon: A small value to avoid division by zero.
    Returns:
        average precision
    """
    # Sort by prediction score if threshold is not None
    sorted_indices = np.argsort(distances)
    relevant_documents = relevant_documents[sorted_indices]

    # Quotient
    quotient = np.sum(relevant_documents)

    if quotient == 0:
        logger.warning("No positive labels in y_true")
        quotient = epsilon

    # Compute average precision
    ## Precision function
    precisions = np.cumsum(relevant_documents) / (np.arange(1, len(relevant_documents) + 1)) * relevant_documents
    return np.sum(precisions) / quotient


def mean_average_precision(
    relevant_documents: np.ndarray, distances: np.ndarray, epsilon: float = 1e-9
) -> tuple[float, float]:
    """Compute mean average precision for a given threshold.

    Args:
        relevant_documents: An array of shape (n_samples, n_documents) containing the
            relevant documents for each sample.
        distances: An array of shape (n_samples, n_documents) containing the distances for each
            document for each sample.
        epsilon: A small value to avoid division by zero.

    Returns:
        tuple[float, float]: mean average precision, standard deviation
    """

    aps = [
        average_precision(relevant_documents[i], distances[i], epsilon=epsilon) for i in range(len(relevant_documents))
    ]
    return np.mean(aps), np.std(aps)


def rank_one(relevant_documents: np.ndarray, scores: np.ndarray) -> float:
    """Compute rank

    Args:
        relevant_documents: true labels
        scores: predicted labels

    Returns:
        float: rank
    """

    sorted_indices = np.argsort(scores)
    relevant_documents = relevant_documents[sorted_indices]

    return relevant_documents.argmax() + 1


def mean_rank_one(relevant_documents: np.ndarray, scores: np.ndarray) -> tuple[float, float, list]:
    """Compute mean rank

    Args:
        relevant_documents: true labels
        scores: predicted labels

    Returns:
        tuple[float, float, list]: mean rank, standard deviation, list of ranks
    """

    ranks = [rank_one(relevant_documents[i], scores[i]) for i in range(len(relevant_documents))]

    return np.mean(ranks), np.std(ranks), ranks


def mean_reciprocal_rank(relevant_documents: np.ndarray, scores: np.ndarray) -> tuple[float, float, list]:
    """Compute mean reciprocal rank

    Args:
        relevant_documents: true labels
        scores: predicted labels

    Returns:
        tuple[float, float, list]: mean reciprocal rank, standard deviation, list of ranks
    """
    ranks = [1 / rank_one(relevant_documents[i], scores[i]) for i in range(len(relevant_documents))]

    return np.mean(ranks), np.std(ranks), ranks


def precision_at_k(relevant_documents: np.ndarray, scores: np.ndarray, k: int = 1) -> tuple[float, float]:
    """Compute precision at k.

    Args:
        relevant_documents: An array of shape (n_samples, n_documents) containing the
            relevant documents for each sample.
        scores: An array of shape (n_samples, n_documents) containing the scores for each
            document for each sample.
        k: The number of documents to consider.

    Returns:
        tuple[float, float]: mean precision, standard deviation

    """

    precision_scores = []

    for i in range(len(relevant_documents)):
        sorted_indices = np.argsort(scores[i])
        sorted_relevant_documents = relevant_documents[i][sorted_indices]

        precision_scores.append(np.sum(sorted_relevant_documents[:k]) / k)

    return np.mean(precision_scores), np.std(precision_scores)


def calculate_metric_bleu(df_sample, column: str, metric: str = "bleu"):
    bleu = evaluate.load(metric)

    references = df_sample["lyrics"].apply(lambda x: [x.replace("\n", " ")]).tolist()
    predictions = df_sample[column].tolist()

    return bleu.compute(predictions=predictions, references=references)


def calculate_wer(df_sample, column: str):
    wer = evaluate.load("wer")

    references = df_sample["lyrics"].tolist()
    predictions = df_sample[column].tolist()

    return wer.compute(predictions=predictions, references=references)
