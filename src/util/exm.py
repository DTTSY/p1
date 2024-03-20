import time


def tiktok(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        print(f'f name: {func.__name__}; Time elapsed: {end - start}')
        return val
    return wrapper
