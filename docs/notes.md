
We also need consistency, so maybe store the channel grid as an asset, and reuse. 

'''Do you think it would be good to do a weekly emmissions calibration.
Should we do a backfill using historical data for each week in 2025 to now?
so this would be a weekly partitioned dataset of the emmissions.



Weekly cadence makes sense — emissions likely drift on seasonal/rainfall timescales,
 so nightly is overkill and weekly aligns with the 7-day rolling window already in the solver.
 A 2025→now backfill would give a genuine time series of the Q-field you could correlate with rainfall, WWTP outages,
  and dry/wet season shifts — main tradeoff is cost (residence-time particles per qualifying event × 3 sensors per week,
   though the S_row_cache makes reruns cheap) and the risk that some weeks have too few ≥30 ppb events
    to constrain the NNLS at all. Want me to sketch a WeeklyPartitionsDefinition version of the calibration job
     and add a "skip week if < N events" gate,
     or first audit 2025 obs to see how many weeks actually have enough signal?
'
