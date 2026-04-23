# H2S Accuracy Reports

Monthly Quarto report and weekly Slack scorecard for the Tijuana River
Valley H2S forecast. Both artifacts consume the rollup JSON produced by
`accuracy_reporting_job` in
`projects/h2s/src/h2s/defs/accuracy_reporting_pipeline.py`:

```
{ACCURACY_URL}/rolling/{7d,30d,90d}/scorecard.json
{ACCURACY_URL}/monthly/{YYYY-MM}/scorecard.json
{ACCURACY_URL}/alert_performance/30d.json
{ACCURACY_URL}/latest.json
```

`ACCURACY_URL` defaults to the public MinIO URL:
`https://oss.resilientservice.mooo.com/resilentpublic/latest/tijuana/forecast/accuracy_reports`.

## Monthly report

The Quarto template lives next to this README as `monthly_accuracy.qmd`.
Render locally:

```sh
quarto render monthly_accuracy.qmd --to html --output-dir _site
# Or for a PDF:
quarto render monthly_accuracy.qmd --to pdf
```

In production, the `monthly_accuracy_report_html` Dagster asset renders
the template and uploads the HTML to S3. Publishing to Netlify is handled
by the generic Slack-approval webhook in
`resilient_workflows_public/netlify_triggers/`.

## Weekly Slack scorecard

The module at `h2s/reporting/weekly_scorecard.py` reads the 7-day and
30-day rollups and posts a Block Kit card to Slack. Run manually:

```sh
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/... \
    python -m h2s.reporting.weekly_scorecard
```

In production, the `weekly_scorecard_post` Dagster asset wraps the module
and is fired by a weekly schedule.
