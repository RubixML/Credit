<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Strategies\Prior;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\Persisters\Filesystem;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true))
    ->apply(new LambdaFunction(function (array &$sample) {
        if ($sample[2] === '0') {
            $sample[2] = '?';
        }
    }))
    ->apply(new MissingDataImputer(categorical: new Prior()))
    ->apply(new FloatTypeConverter())
    ->apply(new OneHotEncoder())
    ->apply(new FloatTypeConverter())
    ->apply(new ZScaleStandardizer());

[$training, $testing] = $dataset->stratifiedSplit(0.8);

$estimator = new LogisticRegression(128, new Stochastic(new StepDecay(0.01, 100)));

$estimator->setLogger($logger);

$estimator->train($training);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress());

$logger->info('Progress saved to progress.csv');

$report = new AggregateReport([
    new MulticlassBreakdown(),
    new ConfusionMatrix(),
]);

$logger->info('Making predictions');

$predictions = $estimator->predict($testing);

$results = $report->generate($predictions, $testing->labels());

echo $results;

$results->toJSON()->saveTo(new Filesystem('report.json'));

$logger->info('Report saved to report.json');
