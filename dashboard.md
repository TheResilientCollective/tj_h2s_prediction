# Panel with Holovix Dashboard Prompt

Create a clean, minimalist Panel Holoviz dashboard in Python.

---

## Dashboard Requirements

### Layout
- **Top row:**  
  - Left:  map showing the current status of the H2s monitoring stations in the Tijuana River region 
  - Right: polarPlot OpenAir plotting package for the latest 12 hours for each site like Fig. 2. in Heavily polluted Tijuana River drives regional air quality crisis 
- **Bottom row:**  
  - A Timeseries range tool . 
     - A stacked bar chart of the aggregated weekly counts of h2s hazardous (yellow >5 <30, orange >30)
     - weekly complaints bar chart
     - Daily peak streamflow colored by hazard color of the h2s
- Do **not** include titles on the dashboard or on any visualizations.  
- Keep the design simple, compact, and visually consistent.

---

## Data
-  Use actual data from https://oss.resilientservice.mooo.com/resilentpublic/latest/tijuana/forecast_data/modeldata_h2s.parquet 
-  Location For the three sites https://oss.resilientservice.mooo.com/resilenresilentpublic/latest/tijuana/forecast_data/h2s_locations.csv:   
- complaints https://oss.resilientservice.mooo.com/resilentpublic/tijuana/sd_complaints/output/latest
- Use `pills` for sites types, with **all pills active by default**, selection_mode multi.  
- Apply a color scheme consistent with the provided layout.

---

## Sidebar Filters
Include **three filters** on the sidebar:  

1. **Year:** slider to select a single year  
2. **District:** single-select dropdown with an **"All"** option  
3. **Neighborhood:** single-select dropdown with an **"All"** option  

---

## Technical Requirements
- Use **Panel**, **Python**, and **Holoviz**.  
- Prefer **native Panel components** over custom CSS.  
- Required visualizations:  
  -  map  
  - TimeseriesRange 
  - Bar chart  
- Include a placeholder for the logo (image will be added later). 
- The size of the dashboard: 1720×1080.


Background document:
* [Heavily polluted Tijuana River drives regional air quality crisis ](background/science.adv1343.pdf)
