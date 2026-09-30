<?php

/**
 * Renders an interactive scatterplot of the 2-D t-SNE embeddings in
 * embedding.csv, colored by the sample label (yes / no).
 *
 * The result is a single self-contained HTML file (embedding.html) that only
 * needs plotly.js at runtime, loaded from a CDN. Run via `composer plot`.
 */

include __DIR__ . '/vendor/autoload.php';

const EMBEDDING_CSV = __DIR__ . '/embedding.csv';
const OUTPUT_HTML   = __DIR__ . '/embedding.html';

if (!is_file(EMBEDDING_CSV)) {
    fwrite(STDERR, 'embedding.csv not found. Run `composer explore` first.' . PHP_EOL);
    exit(1);
}

$yes = ['x' => [], 'y' => []];
$no  = ['x' => [], 'y' => []];

$lines = file(EMBEDDING_CSV, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);

foreach ($lines as $line) {
    $parts = explode(',', $line);
    if (count($parts) < 3) {
        continue;
    }

    $x = (float) $parts[0];
    $y = (float) $parts[1];
    $label = trim($parts[2]);

    // Only two labels are expected in this dataset.
    if ($label === 'yes') {
        $yes['x'][] = $x;
        $yes['y'][] = $y;
    } elseif ($label === 'no') {
        $no['x'][] = $x;
        $no['y'][] = $y;
    }
}

$marker = ['symbol' => 'circle', 'size' => 6, 'opacity' => 0.45];

$traces = [
    [
        'type' => 'scattergl',
        'name' => 'yes',
        'x' => $yes['x'],
        'y' => $yes['y'],
        'mode' => 'markers',
        'marker' => $marker + ['color' => '#e4572e'],
        'hovertemplate' => 'yes<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
    ],
    [
        'type' => 'scattergl',
        'name' => 'no',
        'x' => $no['x'],
        'y' => $no['y'],
        'mode' => 'markers',
        'marker' => $marker + ['color' => '#4393c3'],
        'hovertemplate' => 'no<br>x=%{x:.2f}<br>y=%{y:.2f}<extra></extra>',
    ],
];

$layout = [
    'title' => [
        'text' => 't-SNE embedding of credit default samples',
        'font' => ['size' => 20],
    ],
    'xaxis' => [
        'title' => 't-SNE dim 1',
        'zeroline' => false,
        'gridcolor' => '#eaeaf0',
    ],
    'yaxis' => [
        'title' => 't-SNE dim 2',
        'zeroline' => false,
        'gridcolor' => '#eaeaf0',
    ],
    'legend' => [
        'title' => ['text' => 'default'],
        'orientation' => 'h',
        'y' => 1.1,
        'x' => 0.5,
    ],
    'plot_bgcolor' => 'rgba(0,0,0,0)',
    'paper_bgcolor' => 'rgba(0,0,0,0)',
    'margin' => ['t' => 60, 'b' => 50, 'l' => 60, 'r' => 40],
];

$config = [
    'responsive' => true,
    'displaylogo' => false,
    'modeBarButtonsToAdd' => ['select2d', 'lasso2d'],
];

$tracesJson   = json_encode($traces, JSON_UNESCAPED_SLASHES);
$layoutJson   = json_encode($layout, JSON_UNESCAPED_SLASHES);
$configJson   = json_encode($config, JSON_UNESCAPED_SLASHES);
$total        = count($yes['x']) + count($no['x']);
$subtext      = sprintf(
    '%d samples · %d defaulted (yes) · %d not defaulted (no)',
    $total,
    count($yes['x']),
    count($no['x'])
);

$html = <<<HTML
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>t-SNE embedding — credit default</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js" charset="utf-8"></script>
<style>
    body { margin: 0; padding: 16px 20px; font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif; background: #fff; }
    .header { margin-bottom: 10px; }
    .header h1 { font-size: 1.25rem; margin: 0 0 4px 0; font-weight: 600; }
    .header p { margin: 0; color: #555; font-size: 0.85rem; }
    #plot { width: 100%; height: calc(100vh - 120px); min-height: 480px; }
</style>
</head>
<body>
<div class="header">
    <h1>t-SNE embedding of credit default samples</h1>
    <p id="summary">$subtext</p>
</div>
<div id="plot"></div>
<script>
    var traces = $tracesJson;
    var layout = $layoutJson;
    var config = $configJson;
    Plotly.newPlot('plot', traces, layout, config);
</script>
</body>
</html>
HTML;

file_put_contents(OUTPUT_HTML, $html);

echo "Wrote " . OUTPUT_HTML . " with {$total} samples (yes: " . count($yes['x']) . ', no: ' . count($no['x']) . ")." . PHP_EOL;
echo "Open it in a browser — only plotly.js is fetched from a CDN at runtime." . PHP_EOL;
