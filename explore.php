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

echo 'Loading data into memory ...' . PHP_EOL;

$dataset = Labeled::fromIterator(new CSV('dataset.csv', true))
    ->apply(new NumericStringConverter());

$stats = $dataset->describe();

echo $stats;

$stats->toJSON()->write('stats.json');

echo 'Stats saved to stats.json' . PHP_EOL;

$dataset = $dataset->randomize()->head(2000);

$embedder = new TSNE(2, 20.0, 20);

$embedder->setLogger(new Screen());

echo 'Embedding ...' . PHP_EOL;

$dataset->apply(new OneHotEncoder())
    ->apply(new ZScaleStandardizer())
    ->apply($embedder)
    ->toCSV(['x', 'y', 'label'])
    ->write('embedding.csv');

echo 'Embedding saved to embedding.csv' . PHP_EOL;