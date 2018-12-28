# Credit Card Default Predictor

An example project that predicts the probability of a customer defaulting on their credit card bill next month using a 30,000 sample dataset, [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) estimator, and data transform [Pipeline](https://github.com/RubixML/RubixML#pipeline). We'll also explore the dataset using a manifold learning technique called [t-SNE](https://github.com/RubixML/RubixML#t-sne).

- **Difficulty**: Medium
- **Training time**: Short
- **Memory needed**: < 1G

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

## Presentation Slides
You can refer to the [slide deck](https://docs.google.com/presentation/d/1ZteG0Rf3siS_o-8x2r2AWw95ntcCggmmEHUfwQiuCnk/edit?usp=sharing) that accompanies this example project if you need extra help or need a more in depth look at the math behind Logistic Regression, Gradient Descent, and the Cross Entropy cost function.

## Tutorial
The dataset provided to us contains 30,000 labeled samples from customers of a Taiwanese credit card issuer. Our objective is to train an estimator that predicts the probability of a customer defaulting on their credit card bill next month. Since this is a *binary* classification problem (*will* or *won't* default next month) we can use Rubix's [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) classifier which implements the Probabilistic interface. Logistic Regression is a supervised learner that uses an algorithm called *Gradient Descent* under the hood. Since Logistic Regression is only compatible with continuous features (*ints* and/or *floats*) we will need a [One Hot Encoder](https://github.com/RubixML/RubixML#one-hot-encoder) to convert all the categorical features such as gender, education, and marital status to continuous ones. We'll also demonstrate standardizing using the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) and model persistence using the [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) wrapper.

### Training
Training is the process of feeding data to the learner so that it can build a model of the problem its trying to solve. In Rubix, data is carried in containers called *Datasets*. Let's start by extracting the data from the provided `dataset.csv` file and instantiating a *Labeled* dataset object from it.

> **Note**: The full code for this section can be found in `train.php`.

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

Here we use the PHP League's [CSV reader](https://csv.thephpleague.com/) to extract the data from the CSV file into two iterators - one for the samples and one for the labels. Next, we instantiate the *labeled* dataset object using the `fromIterator()` factory method.

Now let's split the dataset into a *training* and *testing* set so that we can use the testing set to generate a cross validation report later. We choose 80% of the data to be used for training and 20% to be used for testing. In addtion we randomize and stratify the dataset so that each left and right subset contains roughly the same proportion of positive (*will* default) to negative (*won't* default) samples.

```php
list($training, $testing) = $dataset->randomize()->stratifiedSplit(0.80);
```

> **Note**: For a full list of all the operations you can perform on a dataset object refer to the [API Reference](https://github.com/RubixML/RubixML#dataset-objects).

With our datasets ready to go, we now turn our attention to instantiating the estimator object. Estimators are composable like Lego® bricks in Rubix. Therefore, you can build an estimator to suit the exact needs of your use case. As mentioned earlier, we'll need to do some data *preprocessing* before the data gets to the Logistic Regression estimator. In addition, we'd like to be abe to save the trained model so that we can use it again in another process. This is where meta-Estimators come in.

Meta-Estimators are estimators that wrap or manipulate other estimators. For example, [Pipeline](https://github.com/RubixML/RubixML#pipeline) is a meta-Estimator that takes care of applying various transformations to the dataset before it is handed off to the underlying estimator. For our problem, we will need 3 separate transformers in order to get the data in the *shape* we need it. The [Numeric String Converter](https://github.com/RubixML/RubixML#numeric-string-converter) handles converting numeric strings (ex. '17', '2.03241') to their integer and floating point counterparts. The *only* reason why this is necessary is because the CSV reader only recognizes string types. Next we apply a special encoding to the categorical features of the dataset using the [One Hot Encoder](https://github.com/RubixML/RubixML#one-hot-encoder). Finally, we use the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) to center and scale the features such that they have 0 mean and unit variance. The last transformation has been shown to help the learning algorithm converge faster.

Lastly, we'll wrap the entire Pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) meta-Estimator so that we can save and load it from storage later. Persistent Model takes a [Persister](https://github.com/RubixML/RubixML#persisters) as one of it's parameters which links the model to a particular storage location such as a file on the [Filesystem](https://github.com/RubixML/RubixML#filesystem).

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(new Pipeline([
    new NumericStringConverter(),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
], new LogisticRegression(100, new Momentum(0.001), 1e-4)),
    new Filesystem('credit.model')
);
```

Next we define the hyper-parameters of the Logistic Regression estimator in the following order - *batch size*, Gradient Descent *optimizer* and *learning rate*, and *L2 regularization* amount. The default parameters chosen for this project work fairly well and achieve results as good or slightly better than the results in the original paper, however, feel free to experiment and learn. For more information about the hyper-parameters of the Logistic Regression estimator, refer to the [API Reference](https://github.com/RubixML/RubixML#logistic-regression).

Since the Logistic Regression estimator implements the [Verbose](https://github.com/RubixML/RubixML#verbose) interface, we can hand it any [PSR-3](https://www.php-fig.org/psr/psr-3/) compatible logger and it will spit back helpful logging information. For the purposes of this example we will use the Screen logger that comes built-in with Rubix, but there are many other great loggers that you can use such as [Monolog](https://github.com/Seldaek/monolog) or [Analog](https://github.com/jbroadway/analog) to name a couple.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('credit'));
```

Now we are all set to train the estimator. Simply pass the *training* set that we created earlier to the `train()` method on the estimator.

```php
$estimator->train($training);
```

We can call the `steps()` method on the Logistic Regression base estimator from the outer wrapper to output the value of the cost function at each epoch from training. The loss value should be decreasing at each epoch and changes should get smaller the closer the learner is to converging on the minimum of the cost function. You can later plot the cost function by dumping the values to a CSV file and importing them into your favorite plotting software such as [Plotly](https://plot.ly/) or [Tableu](https://public.tableau.com/en-us/s/).

```php
use League\Csv\Writer;

$steps = $estimator->steps();

$writer = Writer::createFromPath('progress.csv', 'w+');
$writer->insertOne(['loss']);
$writer->insertAll(array_map(null, $steps, []));
```

 As you can see, the learner learns quickly at first and then gradually *lands* as it fine tunes the weights of the model for the best setting.

![Cross Entropy Loss](https://github.com/RubixML/Credit/blob/master/docs/images/cross-entropy-loss.png)

Let's see how well our newly trained model does with some data it has never seen before. A cross validation report allows us to get an idea as to how well the model generalizes to the real world. To generate a validation report consisting of a [Confusion Matrix](https://github.com/RubixML/RubixML#confusion-matrix) and [Multiclass Breakdown](https://github.com/RubixML/RubixML#multiclass-breakdown) wrapped in a [report aggregator](https://github.com/RubixML/RubixML#aggregate-report), simply pass it the predictions made from the testing set along with their ground truth labels. Then let's save the report to a JSON file so we can observe the results at any time we want.

```php
$predictions = $estimator->predict($testing);

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));
```

The [Multiclass Breakdown](https://github.com/RubixML/RubixML#multiclass-breakdown) will look something like this ...

```json
{
    "overall": {
        "accuracy": 0.8235294117647058,
        "precision": 0.7688097147528336,
        "recall": 0.6565293379810704,
        "specificity": 0.6565293379810704,
        "negative_predictive_value": 0.7688097147528336,
        "false_discovery_rate": 0.23119028524716645,
        "miss_rate": 0.34347066201892956,
        "fall_out": 0.34347066201892956,
        "false_omission_rate": 0.23119028524716645,
        "f1_score": 0.6831969048799437,
        "mcc": 0.4102516627298001,
        "informedness": 0.31305867596214076,
        "markedness": 0.5376194295056671
    },
}
```

... and the [Confusion Matrix](https://github.com/RubixML/RubixML#confusion-matrix) should look like this.

```json
{
    "no": {
        "no": 4468,
        "yes": 854
    },
    "yes": {
        "no": 205,
        "yes": 474
    }
}
```

Lastly, direct the estimator to prompt us to save the model to storage.

```php
$estimator->prompt();
```

Now that the training script is set up, we can run the program using the [PHP CLI](http://php.net/manual/en/features.commandline.php) (Command Line Interface) in a terminal window.

To run the training script from the project root:
```sh
$ php train.php
```

That's it, if the results of the training session are good (the researchers in the original paper were able to achieve 82% accuracy using their version of Logistic Regression) then save the model and we'll use it to make predictions on some unknown samples in the next section.

### Predicting
Along with the training data, we provide 5 unknown (*unlabeled*) samples that can be used to demonstrate how to make predictions using the estimator we just trained and saved. First, we'll need to load the data from `unkown.csv` into a dataset object just like before but this time we use an *Unlabeled* dataset.

> **Note**: The full code for this section can be found in `predict.php`.

```php
use League\Csv\Reader;
use Rubix\ML\Datasets\Unlabeled;

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

Using the `load()` factory method on Persistent Model we can reconstitute the model from storage. The `load()` method takes an instance of a persister pointing to the model data in storage.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('credit.model'));
```

Then we'll output an array of class probabilities corresponding to the unknown samples and save them to a JSON file. We could also predict just the class outcomes using the `predict()` method if we wanted to, but since we want to be able to measure varying degrees of risk (high, medium, low, etc.) class probabilities make more sense.

```php
$probabilities = $estimator->proba($dataset);

file_put_contents('probabilities.json', json_encode($probabilities, JSON_PRETTY_PRINT));
```

To run the prediction script from the project root:
```sh
$ php predict.php
```

Now take a look at the predictions and observe the outcomes. Notice that out of the 5 samples, one of them should have a higher probability of defaulting than the others.

### Monte Carlo Simulations
Another form of cross validation is the Monte Carlo simulation. The [Monte Carlo](https://github.com/RubixML/RubixML#monte-carlo) validator works by repeatedly sampling training and testing sets from the master dataset and then averaging the validation score of each trained model. The number of simulations and the ratio of training to testing data can be set by the user. The more simulations executed the more precise the validation score will be. We use the [F1 Score](https://github.com/RubixML/RubixML#f1-score) as the metric of validation because it takes into consideration both the precision and recall of the estimator.

As usual, we'll need to start by loading the data into memory. This time we'll need all of the samples from the dataset.

> **Note**: The full code for this section can be found in `validate.php`.

```php
use League\Csv\Reader;
use Rubix\ML\Datasets\Labeled;

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

Then load the estimator from storage and pass it to the validator to test.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\MonteCarlo;
use Rubix\ML\CrossValidation\Metrics\F1Score;

$estimator = PersistentModel::load(new Filesystem('credit.model'));

$validator = new MonteCarlo(10, 0.2, true);

$score = $validator->test($estimator, $dataset, new F1Score());

var_dump($score);
```

You should get something like this.

```sh
float(67.45)
```

To run the validation script from the project root:
```sh
$ php validate.php
```

A good F1 score for this dataset using Logistic Regression is in the higher 0.60s. If you're comfortable so far continue on to the next section where we'll explore a more advanced machine learning concept called manifold learning. We'll use it to visualize the credit card dataset to derive some intuition about the problem.

### Exploring the Dataset
The dataset given to us has 26 dimensions and after one hot encoding it becomes over 50 dimensional. Visualizing this type of high-dimensional data is only possible by reducing the number of dimensions to something that makes sense to plot on a chart (1 - 3 dimensions). Such dimensionality reduction is called *Manifold Learning*. Here we will use a popular manifold learning algorithm called [t-SNE](https://github.com/RubixML/RubixML#t-sne) to help us visualize the data.

As always we start by importing the dataset from its CSV file, but this time we are only going to use 500 random samples. The `head()` method on the dataset object will return the first *n* samples and labels from the dataset in a new dataset object.

> **Note**: The full code for this section can be found in `explore.php`.

```php
use League\Csv\Reader;
use Rubix\ML\Datasets\Labeled;

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

We instantiate the estimator using the same transformer pipeline as before and pass in a logger instance so we can monitor the progress of the embedding in real time. Refer to the [t-SNE documentation](https://github.com/RubixML/RubixML#t-sne) in the API reference for an explanation of the hyper-parameters.

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
], new TSNE(2, 30, 12., 100.0, new Euclidean()));

$estimator->setLogger(new Screen('credit'));
```

Then we train the estimator and use it to generate the low dimensional embedding. Finally, we save the embedding to a CSV file where it can be imported into your plotting software.

```php
use League\Csv\Writer;

$estimator->train(clone $dataset); // Clone dataset since we use it again later to make predictions

$predictions = $estimator->predict($dataset);

$writer = Writer::createFromPath('embedding.csv', 'w+');
$writer->insertOne(['x', 'y']);
$writer->insertAll($predictions);
```

> **Note**: Since we are using a transformer pipeline that modifies the dataset in place, we first clone the dataset to keep an original (untransformed) copy in memory to pass to `predict()`.

Here is an example of what a typical embedding would look like when plotted. As you can see the samples form two distinct blobs that correspond to the group likely to *default* and the group likely to pay *on time*. If you wanted to, you could even plot the labels such that each point is colored accordingly to its class label.

![Example t-SNE Embedding](https://github.com/RubixML/Credit/blob/master/docs/images/t-sne-embedding.png)

To run the embedding script from the project root:
```sh
$ php explore.php
```

> **Note**: Due to the stochastic nature of the t-SNE algorithm, each embedding will look a little different from the last. The important information is contained in the overall *structure* of the data.

### Wrap Up
- [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) is a type of classifier that uses a supervised learning algorithm called Gradient Descent.
- Data is passed around in [Dataset objects](https://github.com/RubixML/RubixML#dataset-objects)
- We can use a data transform [Pipeline](https://github.com/RubixML/RubixML#pipeline) to get the data into the correct shape and format for the underlying estimator to understand
- [Cross Validation](https://github.com/RubixML/RubixML#cross-validation) allows us to test the generalization performance of the trained estimator
- We can convert high-dimensional datasets to easily visualizable low-dimensional represenations using a process called [Manifold Learning](https://github.com/RubixML/RubixML#embedders)

## Original Dataset
Contact: I-Cheng Yeh
Emails: (1) icyeh '@' chu.edu.tw (2) 140910 '@' mail.tku.edu.tw  
Institutions: (1) Department of Information Management, Chung Hua University, Taiwan. (2) Department of Civil Engineering, Tamkang University, Taiwan. other contact information: 886-2-26215656 ext. 3181

### References
[1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.
