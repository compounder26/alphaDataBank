# Modal Handler Architecture Documentation

## Overview
This document outlines the modal handler architecture for the Alpha Dashboard, providing guidelines for maintaining consistency and completeness across all interactive chart components.

## Architecture Principles

### 1. Service Import Management

**Best Practice: Module-Level Imports**
```python
# GOOD: Import at module level
from ..services import get_recommendation_service, get_analysis_service

def callback():
    service = get_recommendation_service()
```

**Avoid: Function-Level Imports**
```python
# BAD: Import inside function (causes scope issues)
def callback():
    from ..services import get_recommendation_service as _get_rec_service
    service = _get_rec_service()
```

**Rationale:** Module-level imports prevent scope issues and improve performance by avoiding repeated imports.

### 2. Chart-to-Modal Coverage Matrix

All interactive charts MUST have corresponding modal handlers. Use this matrix to verify completeness:

| Chart Component ID | Handler Function | Modal Type | Status |
|-------------------|------------------|------------|---------|
| operators-chart | handle_operator_click | detail-modal | ✅ Implemented |
| all-operators-chart | handle_all_operators_click | detail-modal | ✅ Implemented |
| datafields-chart | handle_datafields_chart_click | detail-modal | ✅ Implemented |
| all-datafields-chart | handle_all_datafields_chart_click | detail-modal | ✅ Implemented |
| dataset-chart | handle_dataset_click | detail-modal | ✅ Implemented |
| category-chart | handle_category_click | detail-modal | ✅ Implemented |
| neutralization-pie-chart | handle_neutralization_click | detail-modal | ✅ Implemented |
| neutralization-bar-chart | handle_neutralization_click | detail-modal | ✅ Implemented |
| main-plot | handle_clustering_click | datafield-detail-modal | ✅ Implemented |

### 3. Modal State Management Pattern

**Standard Modal Handler Structure:**
```python
@callback_wrapper.safe_callback(
    [Output("detail-modal", "is_open", allow_duplicate=True),
     Output("detail-modal-title", "children", allow_duplicate=True),
     Output("detail-modal-body", "children", allow_duplicate=True)],
    [Input("chart-id", "clickData"),
     Input("detail-modal-close", "n_clicks")],
    [State("detail-modal", "is_open"),
     State("analysis-data", "data")],
    prevent_initial_call=True
)
@preserve_prevent_update_logic
def handle_chart_click(chart_click, close_clicks, is_open, analysis_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Handle close action
    if trigger_id == "detail-modal-close":
        return False, "", ""

    # Handle chart click
    if trigger_id == "chart-id" and chart_click:
        try:
            # Extract data from click
            # Generate modal content
            # Return modal state
            return True, title, content
        except Exception as e:
            return True, "Error", create_error_modal_content(str(e))

    raise PreventUpdate
```

### 4. Callback Registration Priority

Callbacks should be registered in this order to avoid conflicts:
1. Core data callbacks (filter, analysis)
2. Chart generation callbacks
3. Modal interaction callbacks
4. Close modal callbacks (last to avoid interference)

### 5. Error Handling Patterns

**Service Call Error Handling:**
```python
try:
    service = get_recommendation_service()
    data = service.get_data()
except Exception as e:
    print(f"Error in service call: {e}")
    return create_error_modal_content(f"Service error: {str(e)}")
```

**Click Data Extraction:**
```python
try:
    item = click_data['points'][0]['y']  # For horizontal bars
    value = click_data['points'][0]['x']
except (KeyError, IndexError, TypeError) as e:
    return True, "Error", create_error_modal_content(f"Invalid click data: {str(e)}")
```

## Testing Strategy

### 1. Component ID Verification
```bash
# Find all chart components
rg "id=['\"].*-chart['\"]" --type py

# Verify each has a handler
rg "Input\(['\"].*-chart['\"]" --type py
```

### 2. Modal Handler Test Matrix

Create a test that clicks each chart programmatically:
```python
def test_all_chart_modals(dash_duo):
    charts = [
        'operators-chart', 'all-operators-chart',
        'datafields-chart', 'all-datafields-chart',
        'dataset-chart', 'category-chart',
        'neutralization-pie-chart', 'neutralization-bar-chart'
    ]

    for chart_id in charts:
        # Simulate click
        # Verify modal opens
        # Verify content loads
        # Close modal
```

### 3. Service Integration Testing
```python
def test_service_availability():
    """Ensure all required services are available."""
    from analysis.dashboard.services import (
        get_analysis_service,
        get_recommendation_service,
        get_clustering_service,
        get_chart_service
    )

    assert get_analysis_service() is not None
    assert get_recommendation_service() is not None
    # etc.
```

## Common Issues & Solutions

### Issue 1: Modal Not Opening on Click
**Symptoms:** Click event fires but modal doesn't open
**Common Causes:**
- Missing `allow_duplicate=True` in Output
- Callback not registered
- Component ID mismatch
- PreventUpdate raised incorrectly

**Solution:** Check callback registration and verify component IDs match

### Issue 2: Import Scope Errors
**Symptoms:** "local variable referenced before assignment"
**Cause:** Function-level imports with aliasing
**Solution:** Use module-level imports

### Issue 3: Multiple Handlers for Same Component
**Symptoms:** Only one handler works
**Cause:** Duplicate Input IDs without proper Output management
**Solution:** Combine handlers or use different modal outputs

### Issue 4: Modal Content Not Loading
**Symptoms:** Modal opens but content is empty
**Common Causes:**
- Service call failure
- Missing data in analysis_data
- Incorrect data extraction from click

**Solution:** Add comprehensive error handling and logging

## Maintenance Checklist

When adding new interactive charts:
1. ✅ Define unique component ID
2. ✅ Create modal content generation function in components module
3. ✅ Add handler in modal_callbacks.py following standard pattern
4. ✅ Register handler in register_modal_callbacks()
5. ✅ Update this documentation's coverage matrix
6. ✅ Add test case for new chart interaction
7. ✅ Verify no duplicate Input IDs
8. ✅ Test error scenarios (null data, service failures)

## Debug Helpers

Add these debug statements when troubleshooting:
```python
print(f"DEBUG: Callback triggered by: {trigger_id}")
print(f"DEBUG: Click data: {click_data}")
print(f"DEBUG: Analysis data keys: {analysis_data.keys() if analysis_data else 'None'}")
```

## Performance Considerations

1. **Service Caching:** Services are singletons - don't recreate
2. **Data Processing:** Process heavy computations in services, not callbacks
3. **Modal Content:** Generate content on-demand, don't pre-compute
4. **Error Recovery:** Always provide fallback content for failures

## Future Improvements

1. **Unified Modal Manager:** Create a single modal manager class
2. **Automatic Handler Generation:** Generate handlers from chart config
3. **Test Automation:** Auto-generate tests from coverage matrix
4. **Error Boundary Component:** Wrap modal content in error boundary
5. **Loading States:** Add loading indicators for slow service calls