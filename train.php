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
use League\Csv\Writer;

use function Rubix\ML\array_transpose;

ini_set('memory_limit', '-1');

echo 'Loading data into memory ...' . PHP_EOL;

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true));
    
$dataset->apply(new NumericStringConverter())
    ->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer());

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new LogisticRegression(128, new StepDecay(0.01, 100));

$estimator->setLogger(new Screen('credit'));

echo 'Training ...' . PHP_EOL;

$estimator->train($training);

$losses = $estimator->steps();

$writer = Writer::createFromPath('progress.csv', 'w+');

$writer->insertOne(['loss']);
$writer->insertAll(array_transpose([$losses]));

echo 'Progress saved to progress.csv' . PHP_EOL;

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

echo 'Making predictions ...' . PHP_EOL;

$predictions = $estimator->predict($testing);

$results = $report->generate($predictions, $testing->labels());

file_put_contents('report.json', json_encode($results, JSON_PRETTY_PRINT));

echo 'Report saved to report.json' . PHP_EOL;
