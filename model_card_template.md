# Model Card
This is a model which aims to predict whether someone's income exceeds $50K based on census data.

## Model Details
This model uses RandomForestClassifier, trained on the Census Income "Adult" dataset from the UCI Machine Learning Repository.

## Intended Use
* The dataset that this model was trained on was created to support the ML community in exploring analysis of ML algorithms. The intended use for this project is for exploratory and learning purposes only, not real financial decisions.

## Training Data
The full dataset (~32561 rows) was split 80/20, and 80% (~26048) of the data was used for training. Categorical features were one-hot encoded and the label binarized.

## Evaluation Data
The data was split 80/20, and 20% of the data was used for evaluation.

## Metrics
On the test set, the model achieved a Precision of 0.7391, a Recall of 0.6384, and an F1(F-beta, beta=1) of 0.6851.

## Ethical Considerations
This dataset is derived from the 1994 Census database and contains sensitive attributes such as race, sex, and native-country. Given the age of the data and the variation in performance across demographic slices(see slice_output.txt), this exercise is more beneficial for learning ML concepts than for informing any world views.

## Caveats and Recommendations
The dataset is imbalanced in that over 75% of entries are reported to make less than or equal to $50K while less than 25% exceed $50K. This class imbalance means the model will tend to under-predict the minority class, and slice metrics will often show weaker recall on small subgroups.