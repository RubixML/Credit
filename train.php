<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\StepDecay;
use Rubix\ML\Other\Loggers\Screen;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Datasets\Unlabeled;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true))
    ->apply(new NumericStringConverter())
    ->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new LogisticRegression(128, new StepDecay(0.01, 100));

$estimator->setLogger($logger);

$estimator->train($training);

$importances = $estimator->featureImportances();

$losses = $estimator->steps();

Unlabeled::build(array_transpose([$losses]))
    ->toCSV(['losses'])
    ->write('progress.csv');

$logger->info('Progress saved to progress.csv');

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);

$results = $report->generate($predictions, $testing->labels());

echo $results;

$results->toJSON()->write('report.json');

$logger->info('Report saved to report.json');
