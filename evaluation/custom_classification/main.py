# Metric submission
import argparse
import numpy as np
import os, json
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from InstructorEmbedding import INSTRUCTOR
from transformers import set_seed
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--cache', type=str, default=None, required=True,
                        help='cache folder path to store models and embeddings')
    parser.add_argument("--prompt", type=str, default='hkunlp/instructor-large')
    parser.add_argument("--seed", type=int, default=42, required=True)
    parser.add_argument("--model_name", default=None, type=str)
    parser.add_argument("--task", default=None, choices=['nyt-location', 'nyt-topic'], type=str)
    parser.add_argument('--output_dir', default=None, type=str)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    definitions = {
        'hkunlp/instructor-base': {
            'nyt-location': 'Represent the sentence for classifying based on locations: ',
            'nyt-topic': 'Represent the sentence for classifying based on topics: ',
        },
        'hkunlp/instructor-large': {
            'nyt-location': 'Represent the sentence for classifying based on locations: ',
            'nyt-topic': 'Represent the sentence for classifying based on topics: ',
        },
        'hkunlp/instructor-xl': {
            'nyt-location': 'Represent the sentence for classifying based on locations: ',
            'nyt-topic': 'Represent the sentence for classifying based on topics: ',
        }
    }
    reference_files = {
        'nyt-location': "/data/dheeraj/OOD/location/df_location.csv",
        'nyt-topic': "/data/dheeraj/OOD/topic/df_topic.csv",
    }
    definition = definitions[args.prompt][args.task]

    df = pd.read_csv(reference_files[args.task])
    texts = df["text"]
    labels = df["label"]

    new_texts = []
    for t in texts:
        new_texts.append(definition + t)

    train_texts, train_labels, test_texts, test_labels = train_test_split(new_texts, labels, train_size=100,
                                                                          random_state=args.seed, shuffle=True,
                                                                          stratify=labels)

    # with FileLock('model.lock'):
    model = INSTRUCTOR(args.model_name, cache_folder=args.cache)
    model.cuda()

    clf = LogisticRegression(
        random_state=args.seed,
        n_jobs=-1,
        max_iter=100,
        verbose=1 if logger.isEnabledFor(logging.DEBUG) else 0,
    )

    X_train = np.asarray(model.encode(train_texts, batch_size=32))
    logger.info(f"Encoding {len(test_texts)} test sentences...")
    # if test_cache is None:
    X_test = np.asarray(model.encode(test_texts, batch_size=32))
    test_cache = X_test
    # else:
    #     X_test = test_cache
    logger.info("Fitting logistic regression classifier...")
    clf.fit(X_train, train_labels)
    logger.info("Evaluating...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average="macro")
    scores = {}
    scores["accuracy"] = accuracy
    scores["f1"] = f1

    print(scores)
    with open(os.path.join(args.output_dir, args.task + ".json"), 'w') as fp:
        json.dump(scores, fp)
