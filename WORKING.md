What This Code Does (Quick Understanding)

1.Downloads stock data

2.Creates features (returns, rolling stats)

3.Injects fake fraud (spikes in price/volume)

4.Uses:
~Isolation Forest (tabular anomaly detection)
~LSTM Autoencoder (time-series anomaly detection)

5.Evaluates using Precision / Recall / F1

6.Visualizes anomalies
