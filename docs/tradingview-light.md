lets make a fastapi server with get 

i want it to be trading view light weight charts

lets use dummy data to feed it, also tell me how should the data look like

the data will be 1H time frame

here are the latest light weight charts docs:

Skip to main content
Getting Started
Tutorials
API Reference
5.1
GitHub

Getting started
Series
Chart types
Price scale
Time scale
Panes
Time zones
Plugins
Overview
Series Primitives
Pane Primitives
Custom Series Types
Canvas Rendering Target
Pixel Perfect Rendering

Migrations
iOS
Android
Release Notes
Series
Version: 5.1
Series
This article describes supported series types and ways to customize them.

Supported types
Area
Series Definition: AreaSeries
Data format: SingleValueData or WhitespaceData
Style options: a mix of SeriesOptionsCommon and AreaStyleOptions
This series is represented with a colored area between the time scale and line connecting all data points:

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const areaSeries = chart.addSeries(AreaSeries, { lineColor: '#2962FF', topColor: '#2962FF', bottomColor: 'rgba(41, 98, 255, 0.28)' });

const data = [{ value: 0, time: 1642425322 }, { value: 8, time: 1642511722 }, { value: 10, time: 1642598122 }, { value: 20, time: 1642684522 }, { value: 3, time: 1642770922 }, { value: 43, time: 1642857322 }, { value: 41, time: 1642943722 }, { value: 43, time: 1643030122 }, { value: 56, time: 1643116522 }, { value: 46, time: 1643202922 }];

areaSeries.setData(data);

chart.timeScale().fitContent();



Bar
Series Definition: BarSeries
Data format: BarData or WhitespaceData
Style options: a mix of SeriesOptionsCommon and BarStyleOptions
This series illustrates price movements with vertical bars. The length of each bar corresponds to the range between the highest and lowest price values. Open and close values are represented with the tick marks on the left and right side of the bar, respectively:

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const barSeries = chart.addSeries(BarSeries, { upColor: '#26a69a', downColor: '#ef5350' });

const data = [
  { open: 10, high: 10.63, low: 9.49, close: 9.55, time: 1642427876 },
  { open: 9.55, high: 10.30, low: 9.42, close: 9.94, time: 1642514276 },
  { open: 9.94, high: 10.17, low: 9.92, close: 9.78, time: 1642600676 },
  { open: 9.78, high: 10.59, low: 9.18, close: 9.51, time: 1642687076 },
  { open: 9.51, high: 10.46, low: 9.10, close: 10.17, time: 1642773476 },
  { open: 10.17, high: 10.96, low: 10.16, close: 10.47, time: 1642859876 },
  { open: 10.47, high: 11.39, low: 10.40, close: 10.81, time: 1642946276 },
  { open: 10.81, high: 11.60, low: 10.30, close: 10.75, time: 1643032676 },
  { open: 10.75, high: 11.60, low: 10.49, close: 10.93, time: 1643119076 },
  { open: 10.93, high: 11.53, low: 10.76, close: 10.96, time: 1643205476 },
  { open: 10.96, high: 11.90, low: 10.80, close: 11.50, time: 1643291876 },
  { open: 11.50, high: 12.00, low: 11.30, close: 11.80, time: 1643378276 },
  { open: 11.80, high: 12.20, low: 11.70, close: 12.00, time: 1643464676 },
  { open: 12.00, high: 12.50, low: 11.90, close: 12.30, time: 1643551076 },
  { open: 12.30, high: 12.80, low: 12.10, close: 12.60, time: 1643637476 },
  { open: 12.60, high: 13.00, low: 12.50, close: 12.90, time: 1643723876 },
  { open: 12.90, high: 13.50, low: 12.70, close: 13.20, time: 1643810276 },
  { open: 13.20, high: 13.70, low: 13.00, close: 13.50, time: 1643896676 },
  { open: 13.50, high: 14.00, low: 13.30, close: 13.80, time: 1643983076 },
  { open: 13.80, high: 14.20, low: 13.60, close: 14.00, time: 1644069476 },
];

barSeries.setData(data);

chart.timeScale().fitContent();


Baseline
Series Definition: BaselineSeries
Data format: SingleValueData or WhitespaceData
Style options: a mix of SeriesOptionsCommon and BaselineStyleOptions
This series is represented with two colored areas between the the base value line and line connecting all data points:

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const baselineSeries = chart.addSeries(BaselineSeries, { baseValue: { type: 'price', price: 25 }, topLineColor: 'rgba( 38, 166, 154, 1)', topFillColor1: 'rgba( 38, 166, 154, 0.28)', topFillColor2: 'rgba( 38, 166, 154, 0.05)', bottomLineColor: 'rgba( 239, 83, 80, 1)', bottomFillColor1: 'rgba( 239, 83, 80, 0.05)', bottomFillColor2: 'rgba( 239, 83, 80, 0.28)' });

const data = [{ value: 1, time: 1642425322 }, { value: 8, time: 1642511722 }, { value: 10, time: 1642598122 }, { value: 20, time: 1642684522 }, { value: 3, time: 1642770922 }, { value: 43, time: 1642857322 }, { value: 41, time: 1642943722 }, { value: 43, time: 1643030122 }, { value: 56, time: 1643116522 }, { value: 46, time: 1643202922 }];

baselineSeries.setData(data);

chart.timeScale().fitContent();



Candlestick
Series Definition: CandlestickSeries
Data format: CandlestickData or WhitespaceData
Style options: a mix of SeriesOptionsCommon and CandlestickStyleOptions
This series illustrates price movements with candlesticks. The solid body of each candlestick represents the open and close values for the time period. Vertical lines, known as wicks, above and below the candle body represent the high and low values, respectively:

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const candlestickSeries = chart.addSeries(CandlestickSeries, { upColor: '#26a69a', downColor: '#ef5350', borderVisible: false, wickUpColor: '#26a69a', wickDownColor: '#ef5350' });

const data = [{ open: 10, high: 10.63, low: 9.49, close: 9.55, time: 1642427876 }, { open: 9.55, high: 10.30, low: 9.42, close: 9.94, time: 1642514276 }, { open: 9.94, high: 10.17, low: 9.92, close: 9.78, time: 1642600676 }, { open: 9.78, high: 10.59, low: 9.18, close: 9.51, time: 1642687076 }, { open: 9.51, high: 10.46, low: 9.10, close: 10.17, time: 1642773476 }, { open: 10.17, high: 10.96, low: 10.16, close: 10.47, time: 1642859876 }, { open: 10.47, high: 11.39, low: 10.40, close: 10.81, time: 1642946276 }, { open: 10.81, high: 11.60, low: 10.30, close: 10.75, time: 1643032676 }, { open: 10.75, high: 11.60, low: 10.49, close: 10.93, time: 1643119076 }, { open: 10.93, high: 11.53, low: 10.76, close: 10.96, time: 1643205476 }];

candlestickSeries.setData(data);

chart.timeScale().fitContent();



Histogram
Series Definition: HistogramSeries
Data format: HistogramData or WhitespaceData
Style options: a mix of SeriesOptionsCommon and HistogramStyleOptions
This series illustrates the distribution of values with columns:

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const histogramSeries = chart.addSeries(HistogramSeries, { color: '#26a69a' });

const data = [{ value: 1, time: 1642425322 }, { value: 8, time: 1642511722 }, { value: 10, time: 1642598122 }, { value: 20, time: 1642684522 }, { value: 3, time: 1642770922, color: 'red' }, { value: 43, time: 1642857322 }, { value: 41, time: 1642943722, color: 'red' }, { value: 43, time: 1643030122 }, { value: 56, time: 1643116522 }, { value: 46, time: 1643202922, color: 'red' }];

histogramSeries.setData(data);

chart.timeScale().fitContent();



Line
Series Definition: LineSeries
Data format: LineData or WhitespaceData
Style options: a mix of SeriesOptionsCommon and LineStyleOptions
This series is represented with a set of data points connected by straight line segments:

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const lineSeries = chart.addSeries(LineSeries, { color: '#2962FF' });

const data = [{ value: 0, time: 1642425322 }, { value: 8, time: 1642511722 }, { value: 10, time: 1642598122 }, { value: 20, time: 1642684522 }, { value: 3, time: 1642770922 }, { value: 43, time: 1642857322 }, { value: 41, time: 1642943722 }, { value: 43, time: 1643030122 }, { value: 56, time: 1643116522 }, { value: 46, time: 1643202922 }];

lineSeries.setData(data);

chart.timeScale().fitContent();



Custom series (plugins)
The library enables you to create custom series types, also known as series plugins, to expand its functionality. With this feature, you can add new series types, indicators, and other visualizations.

To define a custom series type, create a class that implements the ICustomSeriesPaneView interface. This class defines the rendering code that Lightweight Charts™ uses to draw the series on the chart. Once your custom series type is defined, it can be added to any chart instance using the addCustomSeries() method. Custom series types function like any other series.

For more information, refer to the Plugins article.

Customization
Each series type offers a unique set of customization options listed on the SeriesStyleOptionsMap page.

You can adjust series options in two ways:

Specify the default options using the corresponding parameter while creating a series:

// Change default top & bottom colors of an area series in creating time
const series = chart.addSeries(AreaSeries, {
    topColor: 'red',
    bottomColor: 'green',
});

Use the ISeriesApi.applyOptions method to apply other options on the fly:

// Updating candlestick series options on the fly
candlestickSeries.applyOptions({
    upColor: 'red',
    downColor: 'blue',
});

Previous
Getting started
Next
Chart types
Supported types
Area
Bar
Baseline
Candlestick
Histogram
Line
Custom series (plugins)
Customization
Docs
Getting Started
Tutorials
API Reference
Lightweight Charts™ Community
Stack Overflow
Twitter
More
Advanced Charts
TradingView Widgets
Copyright © 2026 TradingView, Inc. Built with Docusaurus.

Skip to main content
Getting Started
Tutorials
API Reference
5.1
GitHub

Getting started
Series
Chart types
Price scale
Time scale
Panes
Time zones
Plugins
Overview
Series Primitives
Pane Primitives
Custom Series Types
Canvas Rendering Target
Pixel Perfect Rendering

Migrations
iOS
Android
Release Notes
Chart types
Version: 5.1
Chart types
Lightweight Charts offers different types of charts to suit various data visualization needs. This article provides an overview of the available chart types and how to create them.

Standard Time-based Chart
The standard time-based chart is the most common type, suitable for displaying time series data.

Creation method: createChart
Horizontal scale: Time-based
Use case: General-purpose charting for financial and time series data
import { createChart } from 'lightweight-charts';

const chart = createChart(document.getElementById('container'), options);

This chart type uses time values for the horizontal scale and is ideal for most financial and time series data visualizations.

const chartOptions = { layout: { textColor: 'black', background: { type: 'solid', color: 'white' } } };
const chart = createChart(document.getElementById('container'), chartOptions);
const areaSeries = chart.addSeries(AreaSeries, { lineColor: '#2962FF', topColor: '#2962FF', bottomColor: 'rgba(41, 98, 255, 0.28)' });

const data = [{ value: 0, time: 1642425322 }, { value: 8, time: 1642511722 }, { value: 10, time: 1642598122 }, { value: 20, time: 1642684522 }, { value: 3, time: 1642770922 }, { value: 43, time: 1642857322 }, { value: 41, time: 1642943722 }, { value: 43, time: 1643030122 }, { value: 56, time: 1643116522 }, { value: 46, time: 1643202922 }];

areaSeries.setData(data);

chart.timeScale().fitContent();



Yield Curve Chart
The yield curve chart is specifically designed for displaying yield curves, common in financial analysis.

Creation method: createYieldCurveChart
Horizontal scale: Linearly spaced, defined in monthly time duration units
Key differences:
Whitespace is ignored for crosshair and grid lines
Specialized for yield curve representation
import { createYieldCurveChart } from 'lightweight-charts';

const chart = createYieldCurveChart(document.getElementById('container'), options);

Use this chart type when you need to visualize yield curves or similar financial data where the horizontal scale represents time durations rather than specific dates.

tip
If you want to spread out the beginning of the plot further and don't need a linear time scale, you can enforce a minimum spacing around each point by increasing the minBarSpacing option in the TimeScaleOptions. To prevent the rest of the chart from spreading too wide, adjust the baseResolution to a larger number, such as 12 (months).

const chartOptions = {
    layout: { textColor: 'black', background: { type: 'solid', color: 'white' } },
    yieldCurve: { baseResolution: 1, minimumTimeRange: 10, startTimeRange: 3 },
    handleScroll: false, handleScale: false,
};

const chart = createYieldCurveChart(document.getElementById('container'), chartOptions);
const lineSeries = chart.addSeries(LineSeries, { color: '#2962FF' });

const curve = [{ time: 1, value: 5.378 }, { time: 2, value: 5.372 }, { time: 3, value: 5.271 }, { time: 6, value: 5.094 }, { time: 12, value: 4.739 }, { time: 24, value: 4.237 }, { time: 36, value: 4.036 }, { time: 60, value: 3.887 }, { time: 84, value: 3.921 }, { time: 120, value: 4.007 }, { time: 240, value: 4.366 }, { time: 360, value: 4.290 }];

lineSeries.setData(curve);

chart.timeScale().fitContent();



Options Chart (Price-based)
The options chart is a specialized type that uses price values on the horizontal scale instead of time.

Creation method: createOptionsChart
Horizontal scale: Price-based (numeric)
Use case: Visualizing option chains, price distributions, or any data where price is the primary x-axis metric
import { createOptionsChart } from 'lightweight-charts';

const chart = createOptionsChart(document.getElementById('container'), options);

This chart type is particularly useful for financial instruments like options, where the price is a more relevant x-axis metric than time.

const chartOptions = {
    layout: { textColor: 'black', background: { type: 'solid', color: 'white' } },
};

const chart = createOptionsChart(document.getElementById('container'), chartOptions);
const lineSeries = chart.addSeries(LineSeries, { color: '#2962FF' });

const data = [];
for (let i = 0; i < 1000; i++) {
    data.push({
        time: i * 0.25,
        value: Math.sin(i / 100) + i / 500,
    });
}

lineSeries.setData(data);

chart.timeScale().fitContent();


Custom Horizontal Scale Chart
For advanced use cases, Lightweight Charts allows creating charts with custom horizontal scale behavior.

Creation method: createChartEx
Horizontal scale: Custom-defined
Use case: Specialized charting needs with non-standard horizontal scales
import { createChartEx, defaultHorzScaleBehavior } from 'lightweight-charts';

const customBehavior = new (defaultHorzScaleBehavior())();
// Customize the behavior as needed

const chart = createChartEx(document.getElementById('container'), customBehavior, options);

This method provides the flexibility to define custom horizontal scale behavior, allowing for unique and specialized chart types.

Choosing the Right Chart Type
Use createChart for most standard time-based charting needs.
Choose createYieldCurveChart when working specifically with yield curves or similar financial data.
Opt for createOptionsChart when you need to visualize data with price as the primary horizontal axis, such as option chains.
Use createChartEx when you need a custom horizontal scale behavior that differs from the standard time-based or price-based scales.
Each chart type provides specific functionality and is optimized for different use cases. Consider your data structure and visualization requirements when selecting the appropriate chart type for your application.

Previous
Series
Next
Price scale
Standard Time-based Chart
Yield Curve Chart
Options Chart (Price-based)
Custom Horizontal Scale Chart
Choosing the Right Chart Type
Docs
Getting Started
Tutorials
API Reference
Lightweight Charts™ Community
Stack Overflow
Twitter
More
Advanced Charts
TradingView Widgets
Copyright © 2026 TradingView, Inc. Built with Docusaurus.

Skip to main content
Getting Started
Tutorials
API Reference
5.1
GitHub

Getting started
Series
Chart types
Price scale
Time scale
Panes
Time zones
Plugins
Overview
Series Primitives
Pane Primitives
Custom Series Types
Canvas Rendering Target
Pixel Perfect Rendering

Migrations
iOS
Android
Release Notes
Price scale
Version: 5.1
Price scale
The price scale (or price axis) is a vertical scale that maps prices to coordinates and vice versa. The conversion rules depend on the price scale mode, the chart's height, and the visible part of the data.

Price scales

Create price scale
By default, a chart has two visible price scales: left and right. Additionally, you can create an unlimited number of overlay price scales, which remain hidden in the UI. Overlay price scales allow series to be plotted without affecting the existing visible scales. This is particularly useful for indicators like Volume, where values can differ significantly from price data.

To create an overlay price scale, assign priceScaleId to a series. Note that the priceScaleId value should differ from price scale IDs on the left and right. The chart will create an overlay price scale with the provided ID.

If a price scale with such ID already exists, a series will be attached to the existing price scale. Further, you can use the provided price scale ID to retrieve its API object using the IChartApi.priceScale method.

See the Price and Volume article for an example of adding a Volume indicator using an overlay price scale.

Modify price scale
To modify the left price scale, use the leftPriceScale option. For the right price scale, use rightPriceScale. To change the default settings for an overlay price scale, use the overlayPriceScales option.

You can use the IChartApi.priceScale method to retrieve the API object for any price scale. Similarly, to access the API object for the price scale that a series is attached to, use the ISeriesApi.priceScale method.

Remove price scale
The default left and right price scales cannot be removed, you can only hide them by setting the visible option to false.

An overlay price scale exists as long as at least one series is attached to it. To remove an overlay price scale, remove all series attached to this price scale.

Previous
Chart types
Next
Time scale
Create price scale
Modify price scale
Remove price scale
Docs
Getting Started
Tutorials
API Reference
Lightweight Charts™ Community
Stack Overflow
Twitter
More
Advanced Charts
TradingView Widgets
Copyright © 2026 TradingView, Inc. Built with Docusaurus.


Skip to main content
Getting Started
Tutorials
API Reference
5.1
GitHub

Getting started
Series
Chart types
Price scale
Time scale
Panes
Time zones
Plugins
Overview
Series Primitives
Pane Primitives
Custom Series Types
Canvas Rendering Target
Pixel Perfect Rendering

Migrations
iOS
Android
Release Notes
Time scale
Version: 5.1
Time scale
Overview
Time scale (or time axis) is a horizontal scale that displays the time of data points at the bottom of the chart.

Time scale

The horizontal scale can also represent price or other custom values. Refer to the Chart types article for more information.

Time scale appearance
Use TimeScaleOptions to adjust the time scale appearance. You can specify these options in two ways:

On chart initialization. To do this, provide the desired options as a timeScale parameter when calling createChart.
On the fly using either the ITimeScaleApi.applyOptions or IChartApi.applyOptions method. Both methods produce the same result.
Time scale API
Call the IChartApi.timeScale method to get an instance of the ITimeScaleApi interface. This interface provides an extensive API for controlling the time scale. For example, you can adjust the visible range, convert a time point or index to a coordinate, and subscribe to events.

chart.timeScale().resetTimeScale();

Visible range
Visible range is a chart area that is currently visible on the canvas. This area can be measured with both data and logical range. Data range usually includes bar timestamps, while logical range has bar indices.

You can adjust the visible range using the following methods:

setVisibleRange
getVisibleRange
setVisibleLogicalRange
getVisibleLogicalRange
Data range
The data range includes only values from the first to the last bar visible on the chart. If the visible area has empty space, this part of the scale is not included in the data range.

Note that you cannot extrapolate time with the setVisibleRange method. For example, the chart does not have data prior 2018-01-01 date. If you set the visible range from 2016-01-01, it will be automatically adjusted to 2018-01-01.

If you want to adjust the visible range more flexible, operate with the logical range instead.

Logical range
The logical range represents a continuous line of values. These values are logical indices on the scale that illustrated as red lines in the image below:

Logical range

The logical range starts from the first data point across all series, with negative indices before it and positive ones after.

The indices can have fractional parts. The integer part represents the fully visible bar, while the fractional part indicates partial visibility. For example, the 5.2 index means that the fifth bar is fully visible, while the sixth bar is 20% visible. A half-index, such as 3.5, represents the middle of the bar.

In the library, the logical range is represented with the LogicalRange object. This object has the from and to properties, which are logical indices on the time scale. For example, the visible logical range on the chart above is approximately from -4.73 to 5.05.

The setVisibleLogicalRange method allows you to specify the visible range beyond the bounds of the available data. This can be useful for setting a chart margin or aligning series visually.

Chart margin
Margin is the space between the chart's borders and the series. It depends on the following time scale options:

barSpacing. The default value is 6.
rightOffset. The default value is 0.
You can specify these options as described above.

Note that if a series contains only a few data points, the chart may have a large margin on the left side.

A series with a few points

In this case, you can call the fitContent method that adjust the view and fits all data within the chart.

chart.timeScale().fitContent();

If calling fitContent has no effect, it might be due to how the library displays data.

The library allocates specific width for each data point to maintain consistency between different chart types. For example, for line series, the plot point is placed at the center of this allocated space, while candlestick series use most of the width for the candle body. The allocated space for each data point is proportional to the chart width. As a result, series with fewer data points may have a small margin on both sides.

Margin

You can specify the logical range with the setVisibleLogicalRange method to display the series exactly to the edges. For example, the code sample below adjusts the range by half a bar-width on both sides.

const vr = chart.timeScale().getVisibleLogicalRange();
chart.timeScale().setVisibleLogicalRange({ from: vr.from + 0.5, to: vr.to - 0.5 });

Previous
Price scale
Next
Panes
Overview
Time scale appearance
Time scale API
Visible range
Data range
Logical range
Chart margin
Docs
Getting Started
Tutorials
API Reference
Lightweight Charts™ Community
Stack Overflow
Twitter
More
Advanced Charts
TradingView Widgets
Copyright © 2026 TradingView, Inc. Built with Docusaurus.

Panes
Panes are essential elements that help segregate data visually within a single chart. Panes are useful when you have a chart that needs to show more than one kind of data. For example, you might want to see a stock's price over time in one pane and its trading volume in another. This setup helps users get a fuller picture without cluttering the chart.

By default, Lightweight Charts™ has a single pane, however, you can add more panes to the chart to display different series in separate areas. For detailed examples and code snippets on how to implement panes in your charts see tutorial.

Customization Options
Lightweight Charts™ offers a few customization options to tailor the appearance and behavior of panes:

Pane Separator Color: Customize the color of the pane separators to match the chart design or improve visibility.

Separator Hover Color: Enhance user interaction by changing the color of separators on mouse hover.

Resizable Panes: Opt to enable or disable the resizing of panes by the user, offering flexibility in how data is displayed.

Managing Panes
While the specific methods to manipulate panes are covered in the detailed example, it's important to note that Lightweight Charts™ provides an API for pane management. This includes adding new panes, moving series between panes, adjusting pane height, and removing panes. The API ensures that developers have full control over the pane lifecycle and organization within their charts.

## Local FastAPI Demo (Candlestick, 1H)

This repository now includes a lightweight FastAPI demo server for TradingView Lightweight Charts:

- Server file: `chart_server.py`
- UI page: `static/index.html`
- API endpoint: `GET /api/candles?limit=200`

### Run

```bash
python -m uvicorn chart_server:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- `http://127.0.0.1:8000/` for the chart page
- `http://127.0.0.1:8000/api/candles?limit=200` for raw JSON

### Data shape for Lightweight Charts CandlestickSeries

For `CandlestickSeries`, each bar must be:

```json
{
  "time": 1739912400,
  "open": 101.2,
  "high": 102.0,
  "low": 100.9,
  "close": 101.7
}
```

Rules:

- `time` is Unix timestamp in **seconds** (UTC), not milliseconds.
- For 1H data, adjacent candle times should differ by `3600`.
- Data should be sorted ascending by `time`.
- OHLC must respect market constraints:
  - `high >= max(open, close)`
  - `low <= min(open, close)`

Array example:

```json
[
  {"time": 1739912400, "open": 101.2, "high": 102.0, "low": 100.9, "close": 101.7},
  {"time": 1739916000, "open": 101.7, "high": 102.3, "low": 101.4, "close": 102.1}
]
```

### MT5 data owned by FastAPI (independent mode)

`chart_server.py` is independent from `bot.py` and pulls market data directly from MT5.

Behavior:

- On startup, FastAPI initializes MT5 using `bot_config.json` (`mt5`, `symbol`).
- It loads last `lookback_candles` closed H1 bars (default 500).
- In a background poller, it refreshes data every 5 seconds by checking for new closed H1 bars.
- `GET /api/candles?limit=...` serves this server-owned MT5 cache.

Recommended startup order:

1. Start chart server: `python -m uvicorn chart_server:app --host 127.0.0.1 --port 8000 --reload`
2. Open chart page: `http://127.0.0.1:8000/`
3. Start bot only if you want live trading: `python bot.py`

### SAMA overlay + signal markers

The chart now overlays a thick SAMA line with color by market state:

- green: bull
- red: bear
- yellow: chop

Long/short flips are also shown as markers:

- long flip: green `L` below bar
- short flip: red `S` above bar

Indicator values are computed server-side using strategy values from `bot_config.json`:

- `ama_length`
- `major_length`
- `minor_length`
- `slope_period`
- `slope_in_range`
- `flat_threshold`

Important: these strategy values are loaded once when `chart_server.py` starts. If you edit `bot_config.json`, restart the server.

Indicator API:

- `GET /api/indicator?limit=300`
- Response shape:

```json
{
  "config": {
    "ama_length": 191,
    "major_length": 28,
    "minor_length": 6,
    "slope_period": 109,
    "slope_in_range": 30.0,
    "flat_threshold": 30.0
  },
  "sama_line": [
    {"time": 1739912400, "value": 101.3, "color": "#22c55e"},
    {"time": 1739916000, "value": 100.9, "color": "#facc15"}
  ],
  "markers": [
    {"time": 1739916000, "position": "belowBar", "shape": "arrowUp", "color": "#22c55e", "text": "L"},
    {"time": 1739923200, "position": "aboveBar", "shape": "arrowDown", "color": "#ef4444", "text": "S"}
  ]
}
```

