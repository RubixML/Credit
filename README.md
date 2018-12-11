# Credit Card Default Example Project

An example Rubix ML project that predicts the probability of a customer defaulting on their credit card bill next month using a Logistic Regression estimator and transform pipeline.

## Installation

Clone the repository locally:
```sh
$ git clone https://github.com/RubixML/Credit
```

Install dependencies:
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

### Slides
Be sure to refer to the [slide deck](https://docs.google.com/presentation/d/1ZteG0Rf3siS_o-8x2r2AWw95ntcCggmmEHUfwQiuCnk/edit?usp=sharing) that accompanies this example project if you need extra help or wanted a more in depth look at the math behind Logistic Regression and Gradient Descent.

## Tutorial
The dataset provided to us contains 30,000 labeled samples from customers of a Taiwanese credit card issuer. Our objective is to train an estimator that predicts the probability of a customer defaulting on their credit card bill next month. Since this is a binary classification problem (*will* or *won't* default) we can use Rubix's [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) classifier which implements the Probabilistic interface. Since Logistic Regression is only compatible with continuous features (*ints* and/or *floats*) we will need to use the [One Hot Encoder](https://github.com/RubixML/RubixML#one-hot-encoder) transformer to convert all the categorical features such as gender, education, and marital status to continuous ones.

### Training
Training is the process of feeding data into the learner so that it can build a model of the problem its trying to solve. In Rubix, data is carried in containers called *Datasets*. Let's start by extracting the dataset from the provided `dataset.csv` file and instantiating a *Labeled* dataset object.

> **Note**: The code for this section can be found in `train.php`.

```php
use Rubix\ML\Datasets\Labeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]);

$labels = $reader->fetchColumn('default');

$dataset = Labeled::fromIterator($samples, $labels);
```

Here we use the PHP League's [CSV reader](https://csv.thephpleague.com/) to extract the data into two iterators - one for the samples and one for the labels. Next, we instantiate a *labeled* dataset object using the `fromIterator()` factory method.

We now turn our attention to instantiating and setting the hyper-parameters of the learner. Since learners are composable like Lego® bricks in Rubix, its easy to rapidly iterate over different models and configurations until you find the best.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\NeuralNet\CostFunctions\CrossEntropy;

$estimator = new PersistentModel(new Pipeline([
    new NumericStringConverter(),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new LogisticRegression(128, new Adam(0.001), 1e-4, 300, 1e-4, new CrossEntropy())),
    new Filesystem(MODEL_FILE)
);
```

Pipeline is a meta-Estimator that takes care of applying various transformations to the dataset before it is handed off to the underlying estimator. For our problem, we will need 3 separate transformers. The [Numeric String Converter](https://github.com/RubixML/RubixML#numeric-string-converter) takes care of converting numeric strings (ex. '17', '2.03241') to their integer and floating point counterparts. The only reason why this is necessary is because the CSV reader only recognizes string types. Next we apply a [One Hot](https://en.wikipedia.org/wiki/One-hot) to the samples. Finally, we use the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) to center and scale the samples such that they have mean 0 and unit variance. The last transformation will help our learning algorithm converge faster.

Next we define the parameters of Logistic Regression in the following order - batch size, optimizer, regularization penalty, max # of training epochs, minimum change in the parameters to continue training, and lastly the cost function. The default parameters chosen for this project are fairly good and achieve results as good or slightly better than the results in the original paper.

Lastly, we wrap the entire Pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) meta-Estimator so that we can save and load it from storage when we need it in another process.

The Logistic Regression estimator implements the Verbose interface and therefore can be handed any PSR-3 compatible logger and it will spit back information. For the purposes of this example we will use the Screen logger that comes with Rubix, however there are many other great loggers out there such as [Monolog](https://github.com/Seldaek/monolog) or [Analog](https://github.com/jbroadway/analog).

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('credit'));
```

Now we are all set to train the estimator. But first, let's split the dataset into a training (80%) and testing set (20%) so that we can use the testing set to generate some reports later. We choose to stratify the dataset so that each set contains roughly the same number of positive (*will* default) to negative (*won't* default) samples.

```php
list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);

$estimator->train($training);
```

To generate the validation report which consists of a [Confusion Matrix](https://github.com/RubixML/RubixML#confusion-matrix) and [Multiclass Breakdown](https://github.com/RubixML/RubixML#multiclass-breakdown) wrapper in a [report aggregator](https://github.com/RubixML/RubixML#aggregate-report) simply pass it the predictions from the testing set along with the ground truth labels.

```php
$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$predictions = $estimator->predict($testing);

$results = $report->generate($predictions, $testing->labels());
```

Lastly, we direct the estimator to prompt us to save the model to storage.

```php
$estimator->prompt();
```

That's it, if the results of the training session are good (the researchers in the original paper were able to achieve 82% accuracy using their version of Logistic Regression) then save the model and we'll use it to make predictions on some unknown samples in the next section.

To run the training script from the project root:
```sh
$ php train.php
```

### Predicting
Along with the training data, we provide 5 unknown (*unlabeled) samples that can be used to demonstrate how to make predictions. We'll need to load the data from `unkown.csv` into a dataset object. This time we use an *Unlabeled* dataset.

```php
use Rubix\ML\Datasets\Unlabeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/unknown.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]);

$dataset = Unlabeled::fromIterator($samples);
```

Now, instead of instantiating the untrained model from scratch, we use Persistent Model to load our trained model from before.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));
```

Finally, we output an array of class probabilities corresponding to the unknown samples and save them to a file in JSON format.

```php
$probabilities = $estimator->proba($dataset);

file_put_contents(PROBS_FILE, json_encode($probabilities, JSON_PRETTY_PRINT));
```

Now take a look at the predictions and observe that out of the 5 samples, one of them should have a higher probability of defaulting than the others.

### Cross Validation
Cross Validation tests the generalization performance of a particular model. There are many forms of cross validation to choose from in Rubix, but for this example we will use Monte Carlo simulations. The [Monte Carlo](https://github.com/RubixML/RubixML#monte-carlo) validator works by repeatedly sampling training and testing sets from the master dataset and averaging the validation score of each model. We use the [F1 Score](https://github.com/RubixML/RubixML#f1-score) as a metric because it takes into consideration both precision and recall of the estimator.

```php
use Rubix\ML\CrossValidation\MonteCarlo;
use Rubix\ML\CrossValidation\Metrics\F1Score;

$validator = new MonteCarlo(10, 0.2, true);

$estimator = PersistentModel::load(new Filesystem(MODEL_FILE));

$score = $validator->test($estimator, $dataset, new F1Score());
```

### Exploring the Dataset
The dataset we use has 26 dimensions and after one hot encoding becomes 50+ dimensions. Visualizing this type of high-dimensional data is only possible by reducing the number of dimensions to something we can plot on a chart (1 - 3 dimensions). Such dimensionality reduction is called *Manifold Learning*. Here we will use a popular manifold learning algorithm called [t-SNE](https://github.com/RubixML/RubixML#t-sne) to help us visualize the data.

As always we start with importing the dataset from CSV, but this time we are only going to use 500 random samples.

```php
use Rubix\ML\Datasets\Labeled;
use League\Csv\Reader;

$reader = Reader::createFromPath(__DIR__ . '/dataset.csv')
    ->setDelimiter(',')->setEnclosure('"')->setHeaderOffset(0);

$samples = $reader->getRecords([
    'credit_limit', 'gender', 'education', 'marital_status', 'age',
    'timeliness_1', 'timeliness_2', 'timeliness_3', 'timeliness_4',
    'timeliness_5', 'timeliness_6', 'balance_1', 'balance_2', 'balance_3',
    'balance_4', 'balance_5', 'balance_6', 'payment_1', 'payment_2',
    'payment_3', 'payment_4', 'payment_5', 'payment_6', 'avg_balance',
    'avg_payment',
]);

$labels = $reader->fetchColumn('default');

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(500);
```

We instantiate the estimator using the same transformer pipeline as before with training and pass in a logger instance so we can monitor the progress of the embedding in real time. Refer to the [t-SNE documentation](https://github.com/RubixML/RubixML#t-sne) in the API reference for an explanation of the hyper-parameters.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Manifold\TSNE;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;

$estimator = new Pipeline([
    new NumericStringConverter(),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new TSNE(2, 30, 12., 100.0, 500, 1e-8, 5, new Euclidean()));

$estimator->setLogger(new Screen('credit'));
```

Then we train the estimator and use it to generate the low dimensional embedding. Finally, we save the embedding to a CSV file where it can be imported into your favorite plotting software such as [Tableu](https://www.tableau.com/) or Excel.

```php
$estimator->train(clone $dataset); // Clone dataset since we use it again later to predict

$predictions = $estimator->predict($dataset);

$writer = Writer::createFromPath(OUTPUT_FILE, 'w+');
$writer->insertOne(['x', 'y']);
$writer->insertAll($predictions);
```

> **Note**: Since we are using a transformer pipeline that modifies the dataset, we first clone the dataset to keep an original (untransformed) copy in memory to pass to `predict()`.

Here is an example of what a typical embedding would look like when plotted. As you can see the samples form two distinct blobs that correspond to the group likely to default and the group likely to pay on time. If you wanted to, you could even plot the labels such that each point is colored accordingly to its class label.

![Example t-SNE Embedding](https://github.com/RubixML/Credit/blob/master/docs/images/t-sne-embedding.png)

> **Note**: Due to the stochastic nature of t-SNE, each embedding will look a little different from the last. The important information is contained in the overall *structure* of the data.

## Original Dataset
Contact: I-Cheng Yeh
Emails: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw  
Institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. other contact information: 886-2-26215656 ext. 3181

### References:
[1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

## License
MIT