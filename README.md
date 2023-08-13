# EmailSpamClassification

Check Threshold for Email Spam Filtering using a simple Python script that extracts spam scores from text files and plots the classification results given the scores.

## Steps

1. Export all spam emails into a long text file including the headers. Do the same for spam files. For example, into files `clean.txt` and `spam.txt`.
2. Install dependencies in `requirements.txt`.
3. Run `python main.py --clean path/to/clean.txt --spam path/to/spam.txt --output path/to/output_directory`
4. Inspect the `statistics.txt` file and the figures in `path/to/output_directory`
