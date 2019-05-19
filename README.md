# Credit Card Default Predictor

An example project that predicts the probability of a customer defaulting on their credit card bill the next month using a 30,000 sample dataset, [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) estimator, and data transform [Pipeline](https://github.com/RubixML/RubixML#pipeline). We'll also explore the dataset using a manifold learning technique called [t-SNE](https://github.com/RubixML/RubixML#t-sne).

- **Difficulty**: Medium
- **Training time**: < 3 Minutes
- **Memory needed**: < 1G

## Installation

Clone the repository locally using [Git](https://git-scm.com/):
```sh
$ git clone https://github.com/RubixML/Credit
```

Install dependencies using [Composer](https://getcomposer.org/):
```sh
$ composer install
```

## Requirements
- [PHP](https://php.net) 7.1.3 or above

## Presentation Slides
You can refer to the [slide deck](https://docs.google.com/presentation/d/1ZteG0Rf3siS_o-8x2r2AWw95ntcCggmmEHUfwQiuCnk/edit?usp=sharing) that accompanies this example project if you need extra help or need a more in depth look at the math behind Logistic Regression, Gradient Descent, and the Cross Entropy cost function.

## Tutorial
The dataset provided to us contains 30,000 labeled samples from customers of a Taiwanese credit card issuer. Our objective is to train an estimator that predicts the probability of a customer defaulting on their credit card bill the next month. Since this is a *binary* classification problem we can use Rubix ML's [Logistic Regression](https://github.com/RubixML/RubixML#logistic-regression) classifier which implements the Probabilistic interface as the estimator. Logistic Regression is a supervised learner that uses an algorithm called *Gradient Descent* under the hood. Since Logistic Regression is only compatible with continuous features (*ints* and/or *floats*) we will need a [One Hot Encoder](https://github.com/RubixML/RubixML#one-hot-encoder) to convert all the categorical features such as gender, education, and marital status to continuous ones. We'll also demonstrate standardizing using the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) and model persistence using the [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) wrapper.

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

> **Note**: For a full list of all the operations you can perform on a dataset object refer to the [API Reference](https://github.com/RubixML/RubixML#dataset-objects).

With our dataset ready to go, we now turn our attention to instantiating the estimator object. As mentioned earlier, we'll need to do some data *preprocessing* before the data gets to the Logistic Regression estimator. In addition, we'd like to be abe to save the trained model so that we can use it again in another process. This is where meta-Estimators come in.

Meta-Estimators are estimators that wrap or manipulate other estimators. For example, [Pipeline](https://github.com/RubixML/RubixML#pipeline) is a meta-Estimator that takes care of applying various transformations to the dataset before it is handed off to the underlying estimator. For our problem, we will need 3 separate transformers in order to get the data in the *shape* we need it. The [Numeric String Converter](https://github.com/RubixML/RubixML#numeric-string-converter) handles converting numeric strings (ex. '17', '2.03241') to their integer and floating point counterparts. The *only* reason why this is necessary is because the CSV reader only recognizes string types. Next we apply a special encoding to the categorical features of the dataset using the [One Hot Encoder](https://github.com/RubixML/RubixML#one-hot-encoder). Finally, we use the [Z Scale Standardizer](https://github.com/RubixML/RubixML#z-scale-standardizer) to center and scale the features such that they have 0 mean and unit variance. The last transformation has been shown to help the learning algorithm converge faster.

Lastly, we'll wrap the entire Pipeline in a [Persistent Model](https://github.com/RubixML/RubixML#persistent-model) meta-Estimator so that we can save and load it from storage later. Persistent Model takes a [Persister](https://github.com/RubixML/RubixML#persisters) as one of it's parameters which links the model to a particular storage location such as a file on the [Filesystem](https://github.com/RubixML/RubixML#filesystem).

```php
use Rubix\ML\Pipeline;
use Rubix\ML\PersistentModel;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\Persisters\Filesystem;

$estimator = new PersistentModel(
    new Pipeline([
        new NumericStringConverter(),
        new OneHotEncoder(),
        new ZScaleStandardizer(),
    ], new LogisticRegression(100, new Adam(0.001))),
    new Filesystem('credit.model')
);
```

Next we define the hyper-parameters of the Logistic Regression estimator in the following order - *batch size*, Gradient Descent *optimizer* and *learning rate*. The default parameters chosen for this tutorial work fairly well and achieve results as good or slightly better than the results attained in the original paper, however, feel free to experiment and learn on your own. For more information about the hyper-parameters of the Logistic Regression estimator, refer to the [API Reference](https://github.com/RubixML/RubixML#logistic-regression).

Since the Logistic Regression estimator implements the [Verbose](https://github.com/RubixML/RubixML#verbose) interface, we can hand it any [PSR-3](https://www.php-fig.org/psr/psr-3/) compatible logger and it will spit back helpful logging information. For the purposes of this example we will use the Screen logger that comes built-in with Rubix, but there are many other great loggers that you can use such as [Monolog](https://github.com/Seldaek/monolog) or [Analog](https://github.com/jbroadway/analog) to name a few.

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('credit'));
```

Now we are all set to train the estimator. Simply pass the *training* set that we created earlier to the `train()` method on the estimator.

```php
$estimator->train($dataset);
```

The `steps()` method on the Logistic Regression base estimator outputs the value of the cost function at each epoch from the last training session. You can later plot the cost function by dumping the values to a CSV file and importing them into your favorite plotting software such as [Plotly](https://plot.ly/) or [Tableu](https://public.tableau.com/en-us/s/).

```php
$steps = $estimator->steps();
```

 The loss value should be decreasing at each epoch and changes should get smaller the closer the learner is to converging on the minimum of the cost function. As you can see, the learner learns quickly at first and then gradually *lands* as it fine tunes the weights of the model for the best setting.

![Cross Entropy Loss](https://raw.githubusercontent.com/RubixML/Credit/master/docs/images/cross-entropy-loss.svg?sanitize=true)

Lastly, we save the trained estimator for later.

```php
$estimator->save();
```

Now that the training script is set up, we can run the program using the [PHP CLI](http://php.net/manual/en/features.commandline.php) (Command Line Interface) in a terminal window.

To run the training script from the project root:
```sh
$ php train.php
```

### Cross Validation
The [Monte Carlo](https://github.com/RubixML/RubixML#monte-carlo) cross validator works by repeatedly sampling training and testing sets from the master dataset and then averaging the validation score of each trained model. The number of simulations and the ratio of training to testing data can be set by the user. The more simulations executed, the more precise the validation score will be. We use the [FBeta](https://github.com/RubixML/RubixML#fbeta) metric to score our estimator because it takes into consideration both the precision and recall of the predictions. As usual, we'll need to start by loading the dataset into memory.

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

Then load the persisted estimator from storage and pass it to the validator to test.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\CrossValidation\MonteCarlo;
use Rubix\ML\CrossValidation\Metrics\FBeta;

$estimator = PersistentModel::load(new Filesystem('credit.model'));

$validator = new MonteCarlo(10, 0.2, true);

$score = $validator->test($estimator, $dataset, new FBeta(1.0));

var_dump($score);
```

You should see a score like this after the testing is complete.

```sh
float(70.45)
```

To run the validation script from the project root:
```sh
$ php validate.php
```

### Predicting
Along with the training data, we provide 5 unknown (*unlabeled*) samples that can be used to demonstrate how to make predictions on new data using the estimator we just trained and persisted. First, we'll need to load the data from `unkown.csv` into a dataset object just like before, but this time we use an *Unlabeled* dataset because we don't know them.

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

Using the static `load()` method on Persistent Model, let's load the trained model into memory from storage.

```php
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;

$estimator = PersistentModel::load(new Filesystem('credit.model'));
```

Then we'll output an array of class probabilities corresponding to the unknown samples and save them to a JSON file. We could also predict just the class outcomes using the `predict()` method if we wanted to, but since we want to be able to measure varying degrees of risk (high, medium, low, etc.) class probabilities make more sense.

```php
$probabilities = $estimator->proba($dataset);
```

To run the prediction script from the project root:
```sh
$ php predict.php
```

### Exploring the Dataset
The dataset given to us has 26 dimensions and after one hot encoding it becomes over 50 dimensional. Visualizing this type of high-dimensional data is only possible by reducing the number of dimensions to something that makes sense to plot on a chart (1 - 3 dimensions). Such dimensionality reduction is called *Manifold Learning*. Here we will use a popular manifold learning algorithm called [t-SNE](https://github.com/RubixML/RubixML#t-sne) to help us visualize the data.

As usual we start by importing the dataset from its CSV file, but this time we are only going to use 1000 of the samples. The `head()` method on the dataset object will return the first *n* samples and labels from the dataset in a new dataset object.

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

$dataset = Labeled::fromIterator($samples, $labels)->randomize()->head(1000);
```

We can perform the necessary transformations on the dataset by passing a Transformer object directly to the `apply()` method on the dataset object. The order in which transformers are applied matters.

```php
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\NumericStringConverter;

$dataset->apply(new NumericStringConverter());
$dataset->apply(new OneHotEncoder());
$dataset->apply(new ZScaleStandardizer());
```

Now we'll instantiate a new t-SNE embedder. Refer to the [t-SNE documentation](https://github.com/RubixML/RubixML#t-sne) in the API reference for a full description of the hyper-parameters.

```php
use Rubix\ML\Manifold\TSNE;

$embedder = new TSNE(2, 30, 12., 100.0);
```

Finally, pass the dataset to the `embed()` method on the [Embedder](https://github.com/RubixML/RubixML#embedders) to return a new low dimensional dataset of the embedding.

```php
$embedding = $embedder->embed($dataset);
```

Here is an example of what a typical embedding looks like when plotted. As you can see the samples form two distinct blobs that correspond to the group likely to *default* and the group likely to pay *on time*. If you wanted to, you could even plot the labels such that each point is colored accordingly to its class label.

![Example t-SNE Embedding](https://raw.githubusercontent.com/RubixML/Credit/master/docs/images/t-sne-embedding.svg?sanitize=true)

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
