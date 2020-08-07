<?php

include __DIR__ . '/vendor/autoload.php';

use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Other\Loggers\Screen;

ini_set('memory_limit', '-1');

$logger = new Screen();

$logger->info('Loading data into memory');

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true))
    ->apply(new NumericStringConverter());

$stats = $dataset->describe();

echo $stats;

$stats->toJSON()->write('stats.json');

$logger->info('Stats saved to stats.json');

$dataset = $dataset->randomize()->head(2000);

$embedder = new TSNE(2, 20.0, 20);

$embedder->setLogger($logger);

$dataset->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer())
    ->apply($embedder)
    ->toCSV(['x', 'y', 'label'])
    ->write('embedding.csv');

$logger->info('Embedding saved to embedding.csv');