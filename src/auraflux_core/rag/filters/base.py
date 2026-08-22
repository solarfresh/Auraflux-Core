from abc import ABC, abstractmethod
from typing import List, Optional, Generic, TypeVar

T = TypeVar('T')


# ----------------------------------------------------------------------
# Base Class: BaseFilter
# ----------------------------------------------------------------------
class BaseFilter(ABC, Generic[T]):
    """
    Abstract Base Class for Chunk Filters.
    Supports both single-item and batch processing modes.
    """

    @abstractmethod
    def passes(self, item: T) -> bool:
        """
        Evaluates a single item.
        Returns True if the item passes the filter, False to discard.
        """
        pass

    def filter_batch(self, items: List[T]) -> List[T]:
        """
        Processes a batch of items.
        Default implementation iterates using `passes`.
        Subclasses can override this method for optimized bulk processing.
        """
        return [item for item in items if self.passes(item)]


# ----------------------------------------------------------------------
# Pipeline Manager: FilterPipeline
# ----------------------------------------------------------------------
class FilterPipeline(Generic[T]):
    """
    Sequential pipeline manager that chains multiple BaseFilters.
    Applies Early-Exit mechanism: discards items immediately upon failing any filter.
    """

    def __init__(self, filters: Optional[List[BaseFilter[T]]] = None):
        self.filters: List[BaseFilter[T]] = filters or []

    def add_filter(self, filter_instance: BaseFilter[T]) -> "FilterPipeline[T]":
        """Adds a filter to the end of the pipeline execution chain."""
        self.filters.append(filter_instance)
        return self

    def process_item(self, item: T) -> Optional[T]:
        """
        Executes all filters sequentially on a single item.
        Returns the item if it passes all filters, or None if dropped (Early-Exit).
        """
        for filter_inst in self.filters:
            if not filter_inst.passes(item):
                return None
        return item

    def process_batch(self, items: List[T]) -> List[T]:
        """
        Executes all filters sequentially over a batch of items.
        Leverages individual `filter_batch` implementations for optimal performance.
        """
        current_items = items
        for filter_inst in self.filters:
            if not current_items:
                break
            current_items = filter_inst.filter_batch(current_items)
        return current_items
