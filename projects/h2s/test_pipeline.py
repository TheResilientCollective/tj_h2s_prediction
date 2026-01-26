"""Test H2S pipeline locally to see errors."""

import dagster as dg
from h2s.defs.defs.h2s_pipeline import (
    raw_environmental_data,
    raw_h2s_actuals,
    merged_h2s_dataset,
    h2s_features,
    h2s_model,
    h2s_predictions,
    predictions_export,
)

# Test materialization
result = dg.materialize(
    assets=[
        raw_environmental_data,
        raw_h2s_actuals,
        merged_h2s_dataset,
        h2s_features,
        h2s_model,
        h2s_predictions,
        predictions_export,
    ],
    raise_on_error=False,
)

print(f"\nMaterialization result: {result.success}")
if not result.success:
    for event in result.all_events:
        if event.event_type_value == "STEP_FAILURE":
            print(f"\nFailure in step: {event.step_key}")
            print(f"Error: {event.event_specific_data}")