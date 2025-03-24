#!/usr/bin/env python3
"""
Decorators for the Retail package.

This module provides decorators for:
- Error handling
- Logging
- Performance timing
- Step execution tracking
"""

import time
import functools
import traceback
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, cast, List, Type, Tuple, Union

from utils.logging_utils import logger, LoggingManager, log_step

# Type variables for better type hinting
F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
R = TypeVar('R')


def log_errors(expected_exceptions: Optional[Union[Type[Exception], List[Type[Exception]]]] = Exception,
               msg: str = "Error in {func_name}",
               reraise: bool = False,
               default_return: Any = None,
               log_args: bool = False) -> Callable[[F], F]:
    """
    Decorator to catch and log exceptions.

    Args:
        expected_exceptions: Exception type or a list of exception types to catch. Defaults to Exception.
        msg: Message template for logging errors. {func_name} will be replaced.
        reraise: Whether to re-raise the exception after logging.
        default_return: Value to return in case of exception (if not re-raising).
        log_args: Whether to log function arguments on error.

    Returns:
        Decorated function that logs errors
    """
    # Ensure expected_exceptions is a tuple
    if not isinstance(expected_exceptions, (list, tuple)):
        exceptions_to_check = (expected_exceptions,)
    else:
        exceptions_to_check = tuple(expected_exceptions)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not any(isinstance(e, exc_type) for exc_type in exceptions_to_check):
                    raise

                # Format error message
                formatted_msg = msg.format(func_name=func_name)

                # Log error with details
                logger.error(f"{formatted_msg}: {str(e)}")

                # Log traceback
                logger.debug("Traceback:\n" + traceback.format_exc())

                # Log arguments if requested
                if log_args:
                    safe_args = [repr(arg) if len(repr(arg)) < 1000 else f"{type(arg).__name__}(size too large)" for arg in args]
                    safe_kwargs = {k: repr(v) if len(repr(v)) < 1000 else f"{type(v).__name__}(size too large)" for k, v in kwargs.items()}
                    logger.debug(f"Function arguments: args={safe_args}, kwargs={safe_kwargs}")

                if reraise:
                    raise
                return default_return

        return cast(F, wrapper)
    return decorator


def timed(*args: Any, log_level: str = "info", step_name: Optional[str] = None) -> Any:
    """
    Decorator to time function execution and log the result.

    Can be used with or without parameters:

    @timed
    def my_func():
        ...
    
    or

    @timed("Model execution")
    def my_func():
        ...

    Args:
        log_level: Logging level to use (debug, info, warning, error).
        step_name: Optional step name. If not provided and decorator is used without arguments, function name is used.

    Returns:
        Decorated function that logs timing information
    """
    # If used as a simple decorator without parameters
    if len(args) == 1 and callable(args[0]):
        f = args[0]
        actual_step_name = step_name or f.__name__
        @functools.wraps(f)
        def wrapper(*w_args: Any, **w_kwargs: Any) -> Any:
            start = time.time()
            try:
                return f(*w_args, **w_kwargs)
            finally:
                elapsed = time.time() - start
                getattr(logger, log_level.lower())(f"{actual_step_name} executed in {elapsed:.2f} seconds")
        return wrapper
    else:
        # Used with parameters, e.g., @timed("Model execution")
        provided_step_name = args[0] if args and isinstance(args[0], str) else step_name
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            actual_step_name = provided_step_name or f.__name__
            @functools.wraps(f)
            def wrapper(*w_args: Any, **w_kwargs: Any) -> Any:
                start = time.time()
                try:
                    return f(*w_args, **w_kwargs)
                finally:
                    elapsed = time.time() - start
                    getattr(logger, log_level.lower())(f"{actual_step_name} executed in {elapsed:.2f} seconds")
            return wrapper
        return decorator


def retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
    retry_message: str = "Retrying {func_name} due to {error} (attempt {attempt}/{max_retries})"
) -> Callable[[F], F]:
    """
    Decorator to retry a function on exception.

    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        exceptions: Exception types to catch and retry on
        on_retry: Optional callback function to call on retry with (exception, attempt) parameters
        retry_message: Message template for logging retries

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            mtries, mdelay = max_retries, delay
            func_name = func.__name__
            
            for attempt in range(1, mtries + 2):  # +2 because range is exclusive and we start at 1
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt > mtries:
                        logger.error(f"Function {func_name} failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Log retry
                    msg = retry_message.format(
                        func_name=func_name, 
                        error=str(e), 
                        attempt=attempt,
                        max_retries=max_retries
                    )
                    logger.warning(msg)
                    
                    # Call on_retry callback if provided
                    if on_retry is not None:
                        on_retry(e, attempt)
                    
                    # Sleep before retry with exponential backoff
                    if attempt < mtries:
                        time.sleep(mdelay)
                        mdelay *= backoff_factor
                        
        return cast(F, wrapper)
    return decorator


def cache_result(
    ttl: Optional[float] = None,
    key_function: Optional[Callable[..., str]] = None,
    max_size: Optional[int] = 1000
) -> Callable[[F], F]:
    """
    Decorator to cache function results in memory.
    
    Args:
        ttl: Time to live for cached items in seconds. None means no expiration.
        key_function: Function to generate cache key from args and kwargs.
                      If None, a default function is used.
        max_size: Maximum number of items to keep in cache. None means no limit.
                      
    Returns:
        Decorated function with caching
    """
    cache: Dict[str, Dict[str, Any]] = {}
    
    def default_key_function(*args: Any, **kwargs: Any) -> str:
        """Generate a default cache key from args and kwargs."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)
    
    def clean_cache(func_name: str) -> None:
        """Clean expired items from cache."""
        if ttl is None:
            return
            
        func_cache = cache.get(func_name, {})
        current_time = time.time()
        expired_keys = [
            key for key, (cached_time, _) in func_cache.items() 
            if current_time - cached_time > ttl
        ]
        
        for key in expired_keys:
            del func_cache[key]
            
        # Enforce max size if needed
        if max_size is not None and len(func_cache) > max_size:
            # Sort by timestamp (oldest first) and remove excess
            sorted_items = sorted(func_cache.items(), key=lambda x: x[1][0])
            excess = len(func_cache) - max_size
            for key, _ in sorted_items[:excess]:
                del func_cache[key]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            key = (key_function or default_key_function)(*args, **kwargs)
            func_name = func.__name__
            func_cache = cache.setdefault(func_name, {})
            
            # Clean expired cache entries periodically
            clean_cache(func_name)
            
            # Check if value is in cache and not expired
            if key in func_cache:
                cached_time, cached_value = func_cache[key]
                if ttl is None or time.time() - cached_time < ttl:
                    logger.debug(f"Cache hit for {func_name}")
                    return cached_value
            
            # Call function and cache result
            result = func(*args, **kwargs)
            func_cache[key] = (time.time(), result)
            logger.debug(f"Cache miss for {func_name}, updated cache")
            return result
            
        return cast(F, wrapper)
    return decorator


# Remove the if __name__ == "__main__" block and test functions 