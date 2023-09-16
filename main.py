import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import List
from sklearn.metrics import roc_auc_score, RocCurveDisplay, confusion_matrix
import numpy as np
import pandas as pd
import codecs

def filter_scores(text: str) -> List[float]:
    # search for all occurances of "X-Spam-Score: {float}\n" in each text file using regex
    # pattern = r"X-Spam-Score:\s+(\d+\.\d+)\n"
    pattern = r"X-Spam-Score:\s+(-?\d+\.\d+)\n"
    # Find all matches of the pattern in the string
    matches = re.findall(pattern, text)
    # Convert the matched strings to float values
    float_values = [float(match) for match in matches]
    return float_values


def main(args):
    # create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    # load text file from from args.clean path
    with codecs.open(args.clean, 'r', encoding="utf-8", errors="ignore") as f:
        clean_mail_txt = f.read()
    # load text file from from args.spam path
    with codecs.open(args.spam, "r", encoding="utf-8", errors="ignore") as f:
        spam_mail_txt = f.read()

    # filter scores from text files
    clean_scores = filter_scores(clean_mail_txt)
    spam_scores = filter_scores(spam_mail_txt)
    both_scores = clean_scores + spam_scores
    N_clean = len(clean_scores)
    N_spam = len(spam_scores)
    N_all = N_clean + N_spam
    is_spam = [0] * N_clean + [1] * N_spam
    statistics = {"Total number of emails: ": N_all, "Number of clean emails: ": N_clean, "Number of spam emails: ": N_spam, "P(Clean): ": N_clean / N_all, "P(Spam): ": N_spam / N_all}
    scores = pd.DataFrame({"Score": both_scores, "Email": is_spam})
    # reassign values Email = "Clean" for 0, "Spam" for 1
    scores["Email"] = scores["Email"].map({0: "Clean", 1: "Spam"})

    # plot kde of scores using seaborn
    plt.clf()
    sns.set_theme(style="whitegrid")
    sns.kdeplot(data=scores, x="Score", hue="Email", fill=True, common_norm=False, palette="crest", alpha=.5, linewidth=0)
    plt.xlabel("Spam score")
    plt.ylabel("Density")
    plt.title("Spam score distribution")
    plt.savefig(os.path.join(args.output, "kde.png"))

    # plot wide histogram of scores
    plt.clf()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(20, 6))
    sns.histplot(data=scores, x="Score", hue="Email", palette="crest", bins=50, alpha=.5, linewidth=0)
    plt.xlabel("Spam score")
    plt.ylabel("Count")
    plt.title("Spam score distribution")
    # add minor ticks to x axis in increments of 0.25
    plt.minorticks_on()
    plt.xticks(np.arange(0, 6.25, 0.25))
    # rotate labels 90 degrees
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(args.output, "histogram.png"))


    aucroc = roc_auc_score(is_spam, both_scores)
    statistics["AUCROC: "] = aucroc
    plt.clf()
    sns.set_theme(style="whitegrid")
    roc_curve = RocCurveDisplay.from_predictions(
        is_spam,
        both_scores,
        name=f"spam vs clean:",
        color="darkorange",
        plot_chance_level=True,
    )
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Spam ROC curves")
    plt.legend()
    plt.savefig(os.path.join(args.output, "roc_curve.png"))

    # list false positives and false negative rates for a range of thresholds from 0 to 6 in 0.1 increments
    thresholds = np.array([0 + i * 0.1 for i in range(61)])
    fpr_list = []
    fnr_list = []
    accuracy = []

    for threshold in thresholds:
        # Convert predicted scores to predicted labels based on the threshold
        predicted_labels = [1 if score >= threshold else 0 for score in both_scores]

        # Compute confusion matrix
        tn, fp, fn, tp = confusion_matrix(is_spam, predicted_labels).ravel()

        # Calculate FNR and FPR
        fnr = fn / (fn + tp)
        fpr = fp / (fp + tn)

        fnr_list.append(fnr)
        fpr_list.append(fpr)
        accuracy.append((tp + tn) / (tp + tn + fp + fn))

    threshold_data = pd.DataFrame({"Threshold": thresholds, "FPR": fpr_list, "FNR": fnr_list, "Accuracy": accuracy})

    # plot x=threshold and y=fals_positives and false_negatives
    plt.clf()
    plt.plot(thresholds, fpr_list, label="FPR")
    plt.plot(thresholds, fnr_list, label="FNR")
    plt.plot(thresholds, accuracy, label="Accuracy")
    # set y ticks to 0.1 increments
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel("Threshold")
    plt.ylabel("False positives/negatives")
    plt.title("False positives and negatives for different thresholds")
    plt.legend()
    plt.savefig(os.path.join(args.output, "false_positives_negatives.png"))

    # save statistics to a text file
    with open(os.path.join(args.output, "statistics.txt"), "w") as f:
        statistics_df = pd.DataFrame.from_dict(statistics, orient="index")
        f.write(statistics_df.to_markdown(tablefmt="grid"))
        f.write(threshold_data.to_markdown(tablefmt="grid", index=False))

    print(f"Statistics saved to {os.path.join(args.output, 'statistics.txt')}")


if __name__ == '__main__':
    # parse two argmuments for file paths: clean and spam

    parser = argparse.ArgumentParser()
    parser.add_argument('--clean', type=str, help='path to clean data')
    parser.add_argument('--spam', type=str, help='path to spam data')
    parser.add_argument('--output', type=str, help='path to output directory')
    args = parser.parse_args()
    main(args)