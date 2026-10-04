"""The shared decode pool."""
import threading
import time

import pytest

import micro_reader
from micro_reader import _pool


@pytest.fixture
def threads():
    old = micro_reader.decode_threads()
    yield micro_reader.set_decode_threads
    micro_reader.set_decode_threads(old)


@pytest.mark.parametrize("n", [1, 3])
def test_sums_task_results(threads, n):
    threads(n)
    assert _pool.run(lambda k=k: k for k in range(100)) == sum(range(100))


def test_bounded_in_flight(threads):
    """The task generator is consumed lazily: at most two tasks per thread
    are created ahead of the finished ones, so whatever tasks hold (bytes,
    decoded segments) stays bounded; and at most one runs per thread."""
    threads(2)
    lock = threading.Lock()
    created = finished = running = peak_running = peak_ahead = 0

    def task():
        nonlocal finished, running, peak_running
        with lock:
            running += 1
            peak_running = max(peak_running, running)
        time.sleep(0.005)
        with lock:
            running -= 1
            finished += 1
        return 1

    def tasks():
        nonlocal created, peak_ahead
        for _ in range(40):
            with lock:
                created += 1
                peak_ahead = max(peak_ahead, created - finished)
            yield task

    assert _pool.run(tasks()) == 40
    assert peak_running <= 2
    assert peak_ahead <= 2 * 2 + 1          # the window, plus the one being submitted


def test_errors_reach_the_caller(threads):
    threads(3)

    def boom():
        raise ValueError("bad segment")

    with pytest.raises(ValueError, match="bad segment"):
        _pool.run([lambda: 1, boom, lambda: 1])


def test_environment_variable(monkeypatch, threads):
    monkeypatch.setattr(_pool, "_threads", None)
    monkeypatch.setenv("MICRO_READER_DECODE_THREADS", "5")
    assert micro_reader.decode_threads() == 5
