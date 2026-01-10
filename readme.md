make_fairface_test_only -> data/fairface_test_only.csv:
Generate csv with test set only

analyze_predictions -> None:
Generates confusion matrix for test_predictions.csv

demographics_deepface -> outputs/deepface_demographics.csv:
Uses deepface to add demographics data to the fer2013 set

merge_demographics -> outputs/merged_predictions_with_demographics.csv:
Merge demographics data with fer2013 set

evaluate_bias_template & evaluate_fairness -> None:
Evaluate model evaluate_bias_template

export_fer_images -> data/fer_images:
From csv to images

train_fer -> test_predictions.csv:
Train cnn on fer2013 without demographic data

vgg_fer -> None:
cnn model

dataset_audit.py -> None:
Information on the dataset
