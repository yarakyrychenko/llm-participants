from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import openai
import time
import warnings
from .utils import MAX_WORKERS

def multithreaded(max_workers=MAX_WORKERS, max_retries=5, backoff_factor=1.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args_list, non_iterables=None, **kwargs_list):
            if non_iterables is None:
                non_iterables = {}
            else:
                non_iterables = dict(non_iterables)

            progress_callback = non_iterables.pop("_progress_callback", None)

            self_obj, *actual_args = args_list
            iterable_lengths = [
                len(arg) for arg in actual_args if isinstance(arg, (list, tuple))
            ]
            iterable_lengths.extend(
                len(value) for value in kwargs_list.values() if isinstance(value, (list, tuple))
            )

            if iterable_lengths:
                n_calls = iterable_lengths[0]
                if any(length != n_calls for length in iterable_lengths):
                    raise ValueError("Iterable multithreaded arguments must have matching lengths.")
            else:
                n_calls = 1

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for idx in range(n_calls):
                    args = [
                        arg[idx] if isinstance(arg, (list, tuple)) else arg
                        for arg in actual_args
                    ]
                    kwargs = {
                        key: (value[idx] if isinstance(value, (list, tuple)) else value)
                        for key, value in kwargs_list.items()
                    }
                    futures.append(
                        executor.submit(
                            retry_request,
                            func,
                            max_retries,
                            backoff_factor,
                            self_obj,
                            *args,
                            **kwargs,
                            **non_iterables,
                        )
                    )

                results = []
                errors = []
                completed = 0
                for future in as_completed(futures):
                    try:
                        results.extend(future.result())
                    except Exception as exc:
                        errors.append(exc)
                    finally:
                        completed += 1
                        if progress_callback is not None:
                            progress_callback(completed, n_calls)
                if errors:
                    summary = "; ".join(
                        f"{type(error).__name__}: {error}" for error in errors[:3]
                    )
                    if len(errors) > 3:
                        summary += f"; ... and {len(errors) - 3} more"
                    warnings.warn(
                        (
                            f"Best-effort parallel execution dropped {len(errors)} "
                            f"threaded task(s): {summary}"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                return results
        return wrapper
    return decorator

def retry_request(func, max_retries, backoff_factor, self_obj, *args, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return func(self_obj, *args, **kwargs)
        except openai.RateLimitError:
            wait_time = backoff_factor ** retries
            print(f'Rate limit exceeded. Retrying in {wait_time:.2f} seconds...')
            time.sleep(wait_time)
            retries += 1
    raise Exception('Max retries exceeded for request.')
