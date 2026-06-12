# task rename models to better reflect forecast

# overview
I want the output naming to better reflect what the model is doing. We want a:
* nowcast, 0-3 hours
* nearcast, 3-6 hours
* forecast, 6-24 hours

# some logic gates
The runs should store the model version (there should be archived versions of the models in s3 so we can repeat the analyses),
## NOWCAST
the XGBoost models for the single site, NESTOR-BES, it participates in all the events, and is often the only site with high H2S
## Nearcast
A forecast model that builds upon itself, but still has the last actual h2s measurment 
the starting point in the nearcast,
but we need to rework the logic to incorporate the forecasted h2s into each hour, 
so it uses the forecasted h2s as features (h2s)
h2s_lag_1h h2s_lag_3h h2s_lag_6h h2s_rolling_6h h2s_rolling_24h
# FORECAST
Same As nearcast, but all forecasted h2s as features.

Models:
* we should be working with two trained models, one with the 33-parameter evidence, and the 19-parameter lean. 
* then we have 3 stations, but focus on nestor-bes as the trigger/first tier warning site.

we should run the nowcast hourly.  

# Alerting:
## Tier 1:
* when the probability of h2s > 0.5 at nestor-bes for the nowcast we run a nearcast.
* we report the nowcast and nearcast. With the estimate number of hours of h2s. the next 6 hours
## Tier 2:
* when the probability of h2s > 0.5 at nestor-bes for the forecast, is >10ppm. we run a forecast.
* we report the forecast.
* we send an email to the slack with future option to send to the health team.
## Tier 3:
* when probability of h2s > 0.5 at nestor-bes for the forecast, > 30ppm
* we send an email to the health team.
* we run all report all three sites

## Alert Performance:
* when the observed h2s > 10ppm at nestor-bes we report that to slack. When the level has dropped below 10ppm for 2 hours we create a report that is send to slack, and stored in s3
* We report how many hours in the past 12 there was h2s > than each of the levels (yellow, orange)
* for each hour forecasted we report the probability of h2s > 0.5 at nestor-bes, and the measured level and compare with the actual 


# Validation:
## Nowcast/Nearcast/Forecast accuracY: 
We store the forecasts for every run. 
When a forecast is run, we compare it to the actual available data. and create a report. 
* We want to add that information to a simple database or just a parquet file in s3. 
* this should be able to rebuild the analysis results that are store in the parquet file.

# training
We want to retrain the models every month, and keep an archive. We can promote a newly trained model from the archive to be used in production.
if it performs better than the previous model. This should be a human in the loop event, so when a model is trained,
we publish the analysis to slack, provide a thought if we should promote the model and instructions to promote

