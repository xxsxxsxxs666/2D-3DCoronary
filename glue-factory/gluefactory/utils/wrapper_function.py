import matplotlib.pyplot as plt
from functools import wraps
import time
from pathlib import Path


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} took ', end_time - start_time, ' s')
        return result

    return wrapper


def select_visualize(default_plt_show=True, default_save_path=None, default_skip=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            plt_show = kwargs.pop('plt_show', default_plt_show)
            save_path = kwargs.pop('save_path', default_save_path)
            skip = kwargs.pop('skip', default_skip)
            if plt_show is False and save_path is None:
                return
            if skip:
                return
            result = func(*args, **kwargs)
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, pad_inches=0)
            if plt_show:
                plt.show()
            plt.close()

            return result
        return wrapper

    return decorator
