import math
from typing import Optional


def quick_sort(a: list) -> list:
    def inner(a: list, start: int, stop: int) -> None:
        if stop - start <= 2:
            median_three_sort(a, start, stop)
            return
        
        piv = partition(a, start, stop)
        inner(a, start, piv - 1)
        inner(a, piv + 1, stop)   
    inner(a, 0, len(a) - 1)
    return a

def median_three_sort(a: list, start: int, stop: int) -> int:
    mid = (start + stop) // 2
    if a[stop] < a[start]:
        a[start], a[stop] = a[stop], a[start]
    
    if a[mid] < a[start]:
        a[start], a[mid] = a[mid], a[start]
    elif a[mid] > a[stop]:
        a[mid], a[stop] = a[stop], a[mid]
    
    return mid


def partition(a: list, start: int, stop: int) -> int:
    piv = median_three_sort(a, start, stop)
    middle_value = a[piv]

    a[piv], a[stop - 1] = a[stop - 1], a[piv]
    piv = stop - 1

    left = start + 1
    right = stop - 2
    while True:
        # Skip values that dont need to swap
        while a[right] > middle_value: right -= 1
        while a[left] < middle_value: left += 1

        if right <= left: break
        
        # Swap values and increment
        a[left], a[right] = a[right], a[left]
        right, left = right - 1, left + 1
    
    # Fix piv to be in the middle
    a[piv], a[left] = a[left], a[piv]
    return left

def quick_sort_dual(a: list) -> list:
    def inner(a: list, start: int, stop: int) -> None:
        if start >= stop:
            return
        if stop - start <= 3:
            median_four_sort(a, start, stop)
            return
    
        piv_1, piv_2 = partition_dual(a, start, stop)
        inner(a, start, piv_1 - 1)
        inner(a, piv_1 + 1, piv_2 - 1)
        inner(a, piv_2 + 1, stop)
    inner(a, 0, len(a) - 1)
    return a


def median_four_sort(a: list, start: int, stop: int) -> tuple[int, int]:
    mid_1 = start + (stop - start) // 3
    mid_2 = start + (2 * (stop - start)) // 3

    # Arrange outer
    if a[start] > a[stop]:
        a[start], a[stop] = a[stop], a[start]
    
    # Arrange inner
    if a[mid_1] > a[mid_2]:
        a[mid_1], a[mid_2] = a[mid_2], a[mid_1]

    # Arrange top
    if a[mid_2] > a[stop]:
        a[mid_2], a[stop] = a[stop], a[mid_2]
    
    # Arrange bottom
    if a[start] > a[mid_1]:
        a[start], a[mid_1] = a[mid_1], a[start]

    # Arrange inner
    if a[mid_1] > a[mid_2]:
        a[mid_1], a[mid_2] = a[mid_2], a[mid_1]

    return mid_1, mid_2


def partition_dual(a: list, start: int, stop: int) -> tuple[int, int]:
    piv_1, piv_2 = median_four_sort(a, start, stop)
    
    # Put pivots at the start and end
    a[piv_1], a[start + 1] = a[start + 1], a[piv_1]
    a[piv_2], a[stop - 1] = a[stop - 1], a[piv_2]
    
    # Update pivot indecies and retrive values
    piv_1, piv_2 = start + 1, stop - 1
    value_1, value_2 = a[piv_1], a[piv_2]

    # Set the pointers
    lo = mid = piv_1 + 1
    hi = piv_2 - 1
    while mid <= hi:
        # This value needs to go to the bottom third
        if a[mid] < value_1:
            a[mid], a[lo] = a[lo], a[mid]
            lo += 1
        # This value needs to go to the upper third
        elif a[mid] >= value_2:
            while a[hi] > value_2 and hi > mid:
                hi -= 1
            a[mid], a[hi] = a[hi], a[mid]
            hi -= 1

            # The new value may need to be moved to the lower third
            if a[mid] < value_1:
                a[mid], a[lo] = a[lo], a[mid]
                lo += 1
        mid += 1
    
    # Put the pivots into the correct places
    lo -= 1
    hi += 1
    a[lo], a[piv_1] = a[piv_1], a[lo]
    a[hi], a[piv_2] = a[piv_2], a[hi]
    return lo, hi
            


def quick_sort_insert_20(a: list):
    def inner(a: list, start: int, stop: int):
        if stop - start <= 20:
            insertion_sort(a, start, stop)
            return
    
        piv_1, piv_2 = partition_dual(a, start, stop)
        inner(a, start, piv_1 - 1)
        inner(a, piv_1 + 1, piv_2 - 1)
        inner(a, piv_2 + 1, stop)
    
    inner(a, 0, len(a) - 1)
    return a

def quick_sort_insert_10(a: list):
    def inner(a: list, start: int, stop: int):
        if stop - start <= 10:
            insertion_sort(a, start, stop)
            return
    
        piv_1, piv_2 = partition_dual(a, start, stop)
        inner(a, start, piv_1 - 1)
        inner(a, piv_1 + 1, piv_2 - 1)
        inner(a, piv_2 + 1, stop)

    inner(a, 0, len(a) - 1)
    return a
    
def insertion_sort(a: list, start: int, end: int) -> list:
    for i in range(start + 1, end + 1):
        for j in range(i, start, -1):
            if a[j-1] <= a[j]: break
            a[j-1], a[j] = a[j], a[j-1]

    return a

sorting_algs = [sorted, quick_sort, quick_sort_dual, quick_sort_insert_10, quick_sort_insert_20]

def _test_sorting_algs_on_list(a: list, algs: list[callable]):
    def list_is_ascending(a: list) -> bool:
        prev = a[0]
        for item in a[1:]:
            if item < prev:
                return False
            prev = item
        return True

    for alg in algs:
        s = sum(a)
        sorted_ = alg(a[:])
        assert list_is_ascending(sorted_)
        assert math.isclose(s, sum(sorted_)) # Rounded to avoid floating point inaccuracies

def test_sort_sorted():
    _test_sorting_algs_on_list([n for n in range(100)], sorting_algs) 

def test_sort_inverted():
    _test_sorting_algs_on_list([-n for n in range(100)], sorting_algs) 

def test_sort_same():
    _test_sorting_algs_on_list([0 for _ in range(100)], sorting_algs) 

def test_sort_random():
    from random import random
    _test_sorting_algs_on_list([random() for _ in range(100)], sorting_algs)

def sort_list_with_algs(a: list, sorting_algs: list[callable], samples: int, desc: Optional[str] = None) -> dict[callable, list[float]]:
    timeings = {alg: list() for alg in sorting_algs}

    n_stop = len(a) + 1
    n_start = n_step = (n_stop - 1) // samples
    ns = [n for n in range(n_start, n_stop, n_step)]
    for n in tqdm(ns, desc=desc):
        for func in timeings.keys():
            timeings[func].append(timeit("func(nums)", setup="nums=a[:n]", number=1, globals=vars()))
    return timeings, ns

from matplotlib.axes import Axes
def plot_timings_on_axes(timeings: dict[callable, list[float]], ns: list[int], axes: Axes, title: Optional[str] = None):
    axes.set_ylabel("Tidsforbruk (ms pr. iterasjon)")
    axes.set_xlabel("Antall tall som blir sortert")
    for times in timeings.values():
        axes.plot(ns, times)

    if title is not None: axes.set_title(title)
    axes.legend([func.__name__ for func in timeings.keys()], loc=0, frameon=True)

def sort_and_plot(a: list, sorting_algs: list[callable], axes: Axes, samples: int = 50, title: Optional[str] = None) -> None:
    timeings, ns = sort_list_with_algs(a, sorting_algs, samples, title)
    plot_timings_on_axes(timeings, ns, axes, title)
    


if __name__=="__main__":
    from timeit import timeit
    from random import random
    from matplotlib import pyplot as plt # For plotting
    from tqdm import tqdm # For loading bar

    max_size = 100000
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    sort_and_plot([random() for _ in range(max_size)], sorting_algs, ax1, title="Random values")
    sort_and_plot([0 for _ in range(max_size)], sorting_algs, ax2, title="Same value")
    sort_and_plot([i for i in range(max_size)], sorting_algs, ax3, title="Already sorted")
    sort_and_plot([-i for i in range(max_size)], sorting_algs, ax4, title="Descending order")

    plt.show()