# Visualization Optimization Strategy

## Philosophy
Clean, professional plots using seaborn defaults with minimal customization for maximum reproducibility and clarity.

## Changes Applied

### Violin Plots (Figures 2 & 3)
- **Before**: `fill=False` (outline only) - looked sparse and weird
- **After**: Default filled violins with `inner='box'` - shows distribution better
- Removed custom colors (COLORS_MODELS) - using seaborn defaults
- Removed grid and despine - cleaner appearance
- Kept `cut=0` to avoid extending beyond data range

### Bar Plots
- **Ensemble plot**: Replaced 'magma' palette (yellow looked bad) with blue gradient `['#4575b4', '#74add1']`
- **Metrics comparison**: Using seaborn defaults instead of custom colors
- **Recovery plot**: Simple blue `#4575b4` with black edges
- Baseline lines: Consistent red `#e74c3c` across all plots

### Scatter Plots  
- Unified color scheme: `#4575b4` (professional blue)
- White edges with `linewidth=0.5` for better point definition
- Alpha `0.6` for balanced visibility
- Identity lines: Gray `#878787` instead of harsh red
- Regression lines: `#e74c3c` (consistent with baselines)

## Color Palette Summary
- **Primary blue**: `#4575b4` (data points, bars)
- **Light blue**: `#74add1` (secondary bars)
- **Red**: `#e74c3c` (baselines, regression, thresholds)
- **Gray**: `#878787` (reference/identity lines)
- **White**: Edge colors for scatter points

## Key Improvements
1. Violins now show data distribution properly (filled vs outline)
2. No yellow (magma) - cleaner blue palette throughout
3. Consistent colors across all figure types
4. Removed over-styling (grids, despine, custom alphas)
5. Based on seaborn defaults = reproducible and familiar
