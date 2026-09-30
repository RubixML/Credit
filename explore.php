<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Loggers\Screen;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Transformers\FloatTypeConverter;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\TSNE;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true));

$dataset = $dataset->apply(new LambdaFunction(function (array &$sample) {
        if ($sample[2] === '0') {
            $sample[2] = '?';
        }
    }))
    ->apply(new MissingDataImputer())
    ->apply(new FloatTypeConverter());

$stats = $dataset->describeByClassLabels();

$stats->toJSON()->saveTo(new Filesystem('stats.json'));

$logger->info('Stats saved to stats.json');

$dataset = $dataset->randomize()->take(2048);

$embedder = new TSNE(
    dimensions: 2,
    rate: 20.0,
    perplexity: 20,
    exaggeration: 12.0,
    epochs: 1000,
    minGradient: 1e-7,
    evalInterval: 10,
    window: 5
);

$embedder->setLogger($logger);

$dataset->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer())
    ->apply($embedder)
    ->exportTo(new CSV('embedding.csv'), overwrite: true);

$logger->info('Embedding saved to embedding.csv');
