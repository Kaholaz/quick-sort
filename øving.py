from functools import reduce

def quick_sort(a: list) -> list:
    quick_sort_inner(a, 0, len(a) - 1)
    return a


def quick_sort_inner(a: list, start: int, stop: int) -> None:
    if stop - start <= 2:
        median_three_sort(a, start, stop)
        return
    
    piv = partition(a, start, stop)
    quick_sort_inner(a, start, piv - 1)
    quick_sort_inner(a, piv + 1, stop)   

def arunan_sort(a: list) -> list:
    def inner(a: list, start, stop):
        if stop - start <= 2:
            median_three_sort(a, start, stop)
            return
    
        piv = partition_arunan(a, start, stop)
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

def partition_arunan(nums, low, high):
    m = median_three_sort(nums, low, high)
    nums[m], nums[high - 1] = nums[high - 1], nums[m]
    pivot = nums[high - 1]

    min_index = low + 1

    for current_index in range(min_index, high - 1):
        if nums[current_index] <= pivot:
            nums[min_index], nums[current_index] = nums[current_index], nums[min_index]
            min_index += 1

    nums[min_index], nums[high - 1] = nums[high - 1], nums[min_index]

    return min_index


def quick_sort_dual(a: list) -> list:
    quick_sort_dual_inner(a, 0, len(a) - 1)
    return a
    
def quick_sort_dual_inner(a: list, start: int, stop: int) -> None:
    if start >= stop:
        return
    if stop - start <= 3:
        median_four_sort(a, start, stop)
        return
    
    piv_1, piv_2 = partition_dual(a, start, stop)
    quick_sort_dual_inner(a, start, piv_1 - 1)
    quick_sort_dual_inner(a, piv_1 + 1, piv_2 - 1)
    quick_sort_dual_inner(a, piv_2 + 1, stop)


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

sorting_algs = [sorted, quick_sort, quick_sort_dual, quick_sort_insert_20, arunan_sort]
def test_sort_sorted():
    a = [i for i in range(100)]
    
    results = [alg(a[:]) for alg in sorting_algs]

    answer = sorted(a[:])
    assert all(answer == result for result in results)

def test_sort_inverted():
    a = [-i for i in range(100)]

    results = [alg(a[:]) for alg in sorting_algs]

    answer = sorted(a[:])
    assert all(answer == result for result in results)

def test_sort_same():
    a = [0 for _ in range(100)]

    results = [alg(a[:]) for alg in sorting_algs]

    answer = sorted(a[:])
    assert all(answer == result for result in results)


def test_sort_random():
    from random import seed, random
    seed(123)
    a = [random() for _ in range(100)]

    results = [alg(a[:]) for alg in sorting_algs]

    answer = sorted(a[:])
    assert all(answer == result for result in results)

if __name__=="__main__":
    from timeit import timeit
    from random import random
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    ns = [n for n in range(2000, 100001, 2000)]

    timeings = {alg: list() for alg in sorting_algs}
    for n in tqdm(ns):
        original_nums = [random() for n in range(n)]

        for func in timeings.keys():
            timeings[func].append(timeit("func(nums)", setup="nums=original_nums[:]", number=1, globals=vars()))
    

    plt.ylabel("Tidsforbruk (ms pr. iterasjon)")
    plt.xlabel("Antall tall som blir sortert")

    for func, times in timeings.items():
        plt.plot(ns, times)
    plt.legend([func.__name__ for func in timeings.keys()], loc=0, frameon=True)
    plt.show()
            