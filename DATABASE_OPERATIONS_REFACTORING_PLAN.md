# Database Operations Refactoring Plan

**Generated**: September 21, 2025
**Status**: ARCHITECT REVIEWED - APPROVED WITH MODIFICATIONS
**Estimated Impact**: 40% code reduction in database operations

## Architect's Overall Assessment

**VERDICT: APPROVED** - This refactoring plan is well-structured and addresses real technical debt. The base class approach (Option 1) is the correct architectural choice for this codebase. However, several modifications are needed to align with existing patterns and minimize risk.

## Executive Summary

The current database operations modules (`database/operations.py` and `database/operations_unsubmitted.py`) contain significant code duplication. This plan outlines a refactoring strategy to consolidate common patterns while maintaining the distinct business logic for submitted vs unsubmitted alphas.

## Current State Analysis

### Code Duplication Issues

**ARCHITECT REVIEW**: ✅ The duplication analysis is accurate. I've confirmed these modules share significant code that violates DRY principles.

**File Comparison**: `database/operations.py` vs `database/operations_unsubmitted.py`

**Identical Imports** (14 duplicates):
```python
# Both files have identical imports
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
import io
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Set
from .schema import get_connection, get_region_id
from sqlalchemy import text
```

**Duplicate Patterns**:
1. **Database Connection Handling** - Same pattern repeated
2. **Transaction Management** - Identical transaction wrapping
3. **Error Handling** - Same exception handling logic
4. **Batch Operations** - Similar batch processing patterns
5. **Region ID Resolution** - Identical `get_region_id()` calls
6. **SQL Parameter Binding** - Same parameter preparation patterns

### Quantified Duplication

**Total Lines**:
- `operations.py`: ~800 lines
- `operations_unsubmitted.py`: ~650 lines
- **Estimated Duplicate Code**: ~320 lines (40% of smaller file)

**Functions with 80%+ Similarity**:
- `insert_alpha()` vs `insert_unsubmitted_alpha()`
- `insert_multiple_pnl_data()` patterns
- `get_all_alpha_ids_by_region()` patterns
- Transaction handling in all functions

## Proposed Refactoring Strategy

### Option 1: Base Class Approach (RECOMMENDED)

**ARCHITECT APPROVAL: ✅ CORRECT CHOICE**

This is the right architectural pattern for this refactoring. The inheritance approach properly separates concerns while maximizing code reuse. However, I recommend the following modifications:

**CRITICAL MODIFICATIONS REQUIRED**:
1. Use ABC (Abstract Base Class) to enforce method contracts
2. Add type hints throughout for better maintainability
3. Include connection pooling management in base class
4. Implement proper logging hierarchy

Create a `BaseOperations` class with common patterns, then inherit for specific behaviors:

```python
# database/base_operations.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
from contextlib import contextmanager

class BaseOperations(ABC):
    """Base class for database operations with common patterns.

    ARCHITECT NOTE: Using ABC ensures proper method implementation in subclasses.
    This prevents silent failures from missing methods.
    """

    def __init__(self, table_prefix: str):
        """
        Initialize with table prefix ('alphas' or 'alphas_unsubmitted').

        Args:
            table_prefix: Table name prefix for this operation type
        """
        self.table_prefix = table_prefix
        self.logger = logging.getLogger(f"{__name__}.{table_prefix}")
        # ARCHITECT: Add connection pool reference from schema.py
        self._connection_pool = None  # Initialize lazily

    @contextmanager
    def execute_with_transaction(self):
        """Common transaction wrapper with error handling.

        ARCHITECT: Use context manager pattern for cleaner transaction handling.
        This ensures proper cleanup even on exceptions.
        """
        conn = None
        try:
            conn = get_connection()
            conn.autocommit = False
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Transaction failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_table_name(self, region: str, table_type: str = "main") -> str:
        """Generate table names consistently."""

    def prepare_alpha_data(self, alpha_data: Dict[str, Any]) -> Dict[str, str]:
        """Common alpha data preparation logic."""

    def bulk_insert_pnl(self, pnl_data: List[Dict], region: str) -> bool:
        """Optimized bulk PNL insertion with common patterns.

        ARCHITECT: This should leverage the existing _optimized functions.
        Don't reinvent the wheel - use execute_values from psycopg2.extras.
        """

    @abstractmethod
    def validate_alpha_data(self, alpha_data: Dict[str, Any]) -> bool:
        """Validate alpha data before insertion.

        ARCHITECT: Each subclass MUST implement validation for their specific rules.
        This ensures business logic differences are properly handled.
        """
        pass

# database/submitted_operations.py
class SubmittedOperations(BaseOperations):
    """Operations for submitted alphas."""

    def __init__(self):
        super().__init__("alphas")

    def insert_alpha(self, alpha_data: Dict[str, Any], region: str):
        """Insert submitted alpha - uses base transaction handling."""

# database/unsubmitted_operations.py
class UnsubmittedOperations(BaseOperations):
    """Operations for unsubmitted alphas."""

    def __init__(self):
        super().__init__("alphas_unsubmitted")

    def insert_alpha(self, alpha_data: Dict[str, Any], region: str):
        """Insert unsubmitted alpha - uses base transaction handling."""
```

### Option 2: Utility Functions Approach

**ARCHITECT VERDICT: ❌ REJECTED**

While utility functions are useful, they don't provide enough structure for this refactoring. The inheritance pattern better models the relationship between submitted and unsubmitted operations. However, we should still create some utility functions for truly generic operations.

Extract common patterns into utility functions:

```python
# database/operation_utils.py
def with_transaction(operation_func):
    """Decorator for transaction management."""

def prepare_sql_params(alpha_data: Dict[str, Any], region: str) -> Dict[str, Any]:
    """Common parameter preparation."""

def execute_bulk_insert(table_name: str, data: List[Dict], conflict_strategy: str):
    """Common bulk insertion logic."""
```

### Option 3: Composition-Based Service

**ARCHITECT VERDICT: ❌ REJECTED**

This approach adds unnecessary complexity. The AlphaDataBank codebase follows a simpler module-based architecture, and introducing service patterns would be inconsistent with the existing design. Keep it simple.

Create a database service with composed utilities:

```python
# database/alpha_service.py
class AlphaDataService:
    """Unified service for alpha data operations."""

    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.query_builder = QueryBuilder()
        self.transaction_manager = TransactionManager()

    def insert_alpha(self, alpha_data: Dict, region: str, is_submitted: bool = True):
        """Single method that handles both submitted and unsubmitted."""
```

## Detailed Implementation Plan

**ARCHITECT MODIFICATION: TIMELINE TOO AGGRESSIVE**

The 4-week timeline is unrealistic given the scope and testing requirements. Recommend 6-8 weeks with proper testing phases.

### Phase 1: Create Base Infrastructure (Weeks 1-2)

**1.1 Create `database/base_operations.py`**
- Extract common transaction patterns
- Implement shared error handling
- Create table name generation utilities
- Add common parameter preparation
- **ARCHITECT ADD**: Create comprehensive unit tests BEFORE refactoring
- **ARCHITECT ADD**: Document all public methods with clear contracts
- **ARCHITECT ADD**: Add performance benchmarks for baseline comparison

**1.2 Create Utility Functions**
- `execute_with_retry()` - Common retry logic
- `prepare_alpha_insert_params()` - Parameter standardization
- `bulk_insert_with_conflict_resolution()` - Optimized bulk operations
- **ARCHITECT ADD**: `sanitize_alpha_code()` - SQL injection prevention
- **ARCHITECT ADD**: `validate_region()` - Ensure region is valid
- **ARCHITECT CRITICAL**: Reuse existing `execute_values` optimization!

**1.3 Add Unit Tests**
- Test transaction rollback scenarios
- Test parameter preparation edge cases
- Test table name generation

### Phase 2: Refactor Submitted Operations (Weeks 3-4)

**ARCHITECT NOTE**: Take more time here. This is the most critical module.

**2.1 Create `database/submitted_operations.py`**
- Inherit from `BaseOperations`
- Implement submitted-specific logic
- Maintain exact same public API

**2.2 Functions to Refactor**:
- `insert_alpha()` - Use base transaction handling
- `insert_multiple_alphas()` - Use base bulk operations
- `insert_pnl_data()` - Use base PNL patterns
- `get_all_alpha_ids_by_region()` - Use base query patterns

**2.3 Validation**
- Run existing scripts to ensure no breaking changes
- Compare database states before/after refactoring
- Performance testing on bulk operations

### Phase 3: Refactor Unsubmitted Operations (Week 5)

**ARCHITECT NOTE**: This should be quick since patterns from Phase 2 apply.

**3.1 Create `database/unsubmitted_operations.py`**
- Follow same pattern as submitted operations
- Handle differences in table schema (no `prod_correlation`)
- Maintain unsubmitted-specific business logic

**3.2 Update Import Statements**
- Update all files importing from `operations.py`
- Update all files importing from `operations_unsubmitted.py`
- Ensure backward compatibility during transition

### Phase 4: Migration and Cleanup (Week 6)

**ARCHITECT CRITICAL**: Add a rollback plan!

**4.1 Migrate Existing Code**
- Update `scripts/run_alpha_databank.py`
- Update `api/alpha_fetcher.py` references
- Update `analysis/` modules that use database operations

**4.2 Remove Old Files**
- **ARCHITECT CHANGE**: DO NOT DELETE - Move to `database/legacy/` directory
- Archive `operations.py` and `operations_unsubmitted.py`
- Update imports across codebase
- Update documentation
- **ARCHITECT ADD**: Keep legacy files for 3 months as fallback

**4.3 Performance Optimization**
- Benchmark new vs old implementation
- Optimize connection pooling in base class
- Add query performance monitoring

## Code Reduction Estimate

### Before Refactoring
```
operations.py:             ~800 lines
operations_unsubmitted.py: ~650 lines
Total:                     1,450 lines
```

### After Refactoring
```
base_operations.py:        ~400 lines (common patterns)
submitted_operations.py:   ~250 lines (submitted-specific)
unsubmitted_operations.py: ~200 lines (unsubmitted-specific)
operation_utils.py:        ~100 lines (utilities)
Total:                     ~950 lines
```

**Reduction**: 500 lines (34.5% reduction)
**Maintenance Benefit**: Single place to fix transaction bugs, connection issues, etc.

## Risk Assessment

**ARCHITECT UPDATE: Some risks are understated**

### LOW RISK ✅
- **Transaction Logic**: Moving to base class is safe - same patterns
- **Connection Management**: Already centralized in `schema.py`
- **Parameter Preparation**: Pure functions, easy to test

### MODERATE RISK ⚠️
- **Table Schema Differences**: Submitted vs unsubmitted have different columns
- **Business Logic Differences**: Different validation rules
- **Import Changes**: Need to update many files
- **ARCHITECT ADD - Performance Risk**: Inheritance may add slight overhead
- **ARCHITECT ADD - Testing Gap**: No existing unit tests to validate against

### MITIGATION STRATEGIES
1. **Incremental Migration**: Refactor one module at a time
2. **Comprehensive Testing**: Test with real database operations
3. **Backward Compatibility**: Keep old files during transition
4. **Database Validation**: Compare before/after database states
5. **ARCHITECT ADD - Feature Flags**: Use environment variable to switch between old/new implementation
6. **ARCHITECT ADD - Monitoring**: Add detailed logging to track any behavioral changes
7. **ARCHITECT ADD - Performance Testing**: Run correlation calculations before/after to ensure no regression
8. **ARCHITECT ADD - Rollback Plan**: Document exact steps to revert if issues arise

## Success Criteria

**ARCHITECT NOTE: Add quantifiable metrics**

### Functional Requirements
✅ All existing functionality preserved
✅ Same performance or better
✅ No breaking changes to public APIs
✅ All existing scripts continue to work

### Quality Requirements
✅ 30%+ reduction in database operation code
✅ Consistent error handling across all operations
✅ Single place to modify transaction logic
✅ Improved test coverage for database operations
✅ **ARCHITECT ADD**: No performance regression (maintain <100ms for single operations)
✅ **ARCHITECT ADD**: Memory usage stays within 10% of current baseline
✅ **ARCHITECT ADD**: All existing scripts pass without modification

### Maintenance Requirements
✅ New database operations easier to implement
✅ Common bugs fixed in one place
✅ Better separation of concerns
✅ Clear inheritance hierarchy

## Alternative Approaches Considered

### 1. Complete Merge into Single Module
**Pros**: Maximum code sharing
**Cons**: Business logic confusion, harder to maintain
**Decision**: Rejected - submitted vs unsubmitted have different business rules
**ARCHITECT AGREES**: ✅ Correct decision. Separation of concerns is critical.

### 2. Keep Current Structure
**Pros**: No migration risk
**Cons**: Continued maintenance of duplicate code
**Decision**: Rejected - technical debt will continue to grow
**ARCHITECT AGREES**: ✅ The duplication is already causing maintenance issues.

### 3. Extract to External Database ORM
**Pros**: Professional ORM features
**Cons**: Major architectural change, learning curve
**Decision**: Rejected - over-engineering for this project scope
**ARCHITECT AGREES**: ✅ SQLAlchemy is already used minimally. Full ORM would conflict with existing raw SQL approach and Cython optimizations.

## Implementation Timeline

**ARCHITECT REVISED TIMELINE**:

**Weeks 1-2**: Base infrastructure and utilities + comprehensive testing framework
**Weeks 3-4**: Submitted operations refactoring with parallel testing
**Week 5**: Unsubmitted operations refactoring
**Week 6**: Migration, performance validation, and controlled rollout
**Week 7-8**: Buffer for issues and optimization

**Total Effort**: 6-8 weeks (ARCHITECT: More realistic)
**Risk Level**: Moderate (ARCHITECT: Upgraded from Low-Moderate)
**Code Reduction**: ~35% (ARCHITECT: Confirmed achievable)
**Maintenance Improvement**: High (ARCHITECT: Primary benefit)

## ARCHITECT'S ADDITIONAL REQUIREMENTS

### Pre-Implementation Checklist
1. ✅ Create full backup of current database
2. ✅ Document current performance baselines
3. ✅ List all scripts/modules that import database operations
4. ✅ Create integration test suite BEFORE refactoring
5. ✅ Get stakeholder approval for extended timeline

### Critical Design Decisions
1. **USE EXISTING PATTERNS**: Follow the connection pooling from schema.py
2. **PRESERVE OPTIMIZATIONS**: Keep all _optimized functions and Cython integration
3. **MAINTAIN COMPATIBILITY**: Public API must remain unchanged
4. **RESPECT REGIONS**: Keep region-specific table handling intact
5. **PROTECT PERFORMANCE**: Bulk operations must maintain current speed

### Post-Implementation Validation
1. Run full alpha fetch for all regions
2. Calculate correlations and compare results
3. Verify dashboard still loads correctly
4. Check memory usage under load
5. Validate transaction rollback scenarios

### Rollback Strategy
If critical issues arise:
1. Revert git commits to previous state
2. Restore imports to use legacy modules
3. Clear any cached analysis data
4. Document issues for future attempt
5. Keep refactored code in feature branch for analysis

---

## FINAL ARCHITECT VERDICT

**APPROVED WITH MODIFICATIONS** ✅

This refactoring is necessary and well-planned. The base class inheritance approach correctly balances code reuse with maintainability. However, the following modifications are MANDATORY:

1. **Extend timeline to 6-8 weeks** - Quality over speed
2. **Add comprehensive testing BEFORE refactoring** - No safety net currently exists
3. **Use ABC for base class** - Enforce contracts properly
4. **Keep legacy files in database/legacy/** - Don't delete, archive
5. **Add feature flags for gradual rollout** - Minimize risk
6. **Preserve ALL existing optimizations** - Especially Cython integration
7. **Add performance benchmarks** - Measure, don't guess

**Key Success Factor**: This refactoring MUST be invisible to all consuming code. If any script needs modification beyond import statements, the refactoring has failed.

**Approved by**: Senior Software Architect
**Date**: September 21, 2025
**Next Step**: Create integration tests before touching any production code