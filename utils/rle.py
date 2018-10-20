"""
from Even Faster RLE
https://www.kaggle.com/danmoller/even-faster-rle
"""

import numpy as np
import multiprocessing
from collections import ChainMap


def RLenc(img, order='F', format=True):
    """
    Source https://www.kaggle.com/bguberfain/unet-with-depth
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs


class Consumer(multiprocessing.Process):
    """Consumer for performing a specific task."""

    def __init__(self, task_queue, result_queue):
        """Initialize consumer, it has a task and result queues."""
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.daemon = True

    def run(self):
        """Actual run of the consumer."""
        #print('start rle consumer {}'.format(self.pid))
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                self.task_queue.task_done()
                break
            # Fetch answer from task
            answer = next_task()
            # Put into result queue
            self.result_queue.put(answer)
            self.task_queue.task_done()
        #print('stop rle consumer {}'.format(self.pid))
        return


class RleTask(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, idx, img):
        """Save image to self."""
        self.img = img
        self.idx = idx

    def __call__(self):
        """When object is called, encode."""
        return {self.idx: RLenc(self.img)}


class FastRle(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()

    def add(self, idx, img):
        """Add a task to perform."""
        self._tasks.put(RleTask(idx, img))

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        while not self._results.empty():
            singles.append(self._results.get())
        return dict(ChainMap({}, *singles))


# even faster RLE encoder
def toRunLength(x, firstDim=2):
    if firstDim == 2:
        x = np.swapaxes(x, 1, 2)

    x = (x > 0.5).astype(int)
    x = x.reshape((x.shape[0], -1))
    x = np.pad(x, ((0, 0), (1, 1)), 'constant')

    x = x[:, 1:] - x[:, :-1]
    starts = x > 0
    ends = x < 0

    rang = np.arange(x.shape[1])

    results = []

    for image, imStarts, imEnds in zip(x, starts, ends):
        st = rang[imStarts]
        en = rang[imEnds]

        #         counts = (en-st).astype(str)
        #         st = (st+1).astype(str)

        #         res = np.stack([st,counts], axis=-1).reshape((-1,))
        #         res = np.core.defchararray.join(" ", res)

        res = ""
        for s, e in zip(st, en):
            res += str(s + 1) + " " + str(e - s) + " "

        results.append(res[:-1])
    # print("called")

    return results


class FasterTask(object):
    """Wrap the RLE Encoder into a Task."""

    def __init__(self, array, startIndex):
        """Save array to self."""
        self.array = array
        self.startIndex = startIndex

    def __call__(self):
        """When object is called, encode."""
        return (toRunLength(self.array), self.startIndex)


class FasterRle(object):
    """Perform RLE in paralell."""

    def __init__(self, num_consumers=2):
        """Initialize class."""
        self._tasks = multiprocessing.JoinableQueue()
        self._results = multiprocessing.Queue()
        self._n_consumers = num_consumers

        # Initialize consumers
        self._consumers = [Consumer(self._tasks, self._results) for i in range(self._n_consumers)]
        for w in self._consumers:
            w.start()
        self.startIndex = 0

    def add(self, array):
        """Add a task to perform."""
        self._tasks.put(FasterTask(array, self.startIndex))
        self.startIndex += len(array)

    def get_results(self):
        """Close all tasks."""
        # Provide poison pill
        [self._tasks.put(None) for _ in range(self._n_consumers)]
        # Wait for finish
        self._tasks.join()
        # Return results
        singles = []
        while not self._results.empty():
            singles.append(self._results.get())

        resultDic = dict()
        for rles, start in singles:
            # print('start:', start)
            for i, rle in enumerate(rles):
                # print('i:', i)
                resultDic[start + i] = rle
        return resultDic


if __name__ == '__main__':
    import timeit

    example_image = np.random.uniform(0, 1, size=(1000, 101, 101)) > 0.5

    # Wrap the FastRle class into a method so we measure the time
    def original(array):
        results = {}
        for i, arr in enumerate(array):
            results[i] = RLenc(arr)
        return results


    def faster(array):
        rle = FastRle(4)
        for i, arr in enumerate(array):
            rle.add(i, arr)
        return rle.get_results()


    def pureNumpy(array):
        rle = toRunLength(array)
        rle = dict(enumerate(rle))
        return rle


    def evenFaster(array):
        # make sure you treat this properly when len(array) % 4 != 0
        rle = FasterRle(4)
        subSize = len(array) // 4

        for i in range(0, len(array), subSize):
            rle.add(array[i:i + subSize])
        return rle.get_results()

    x = original(example_image)
    y = faster(example_image)
    z = pureNumpy(example_image)
    w = evenFaster(example_image)

    # Make sure they are the same
    print("Comparing values:\n")

    #comparison = []
    #for key in x:
    #    comparison.append(x[key] == y[key])
    #print('Original vs Parallel:', np.all(comparison))

    comparison = []
    for key in x:
        comparison.append(x[key] == z[key])
    print("Original vs pure numpy:", np.all(comparison))

    comparison = []
    for key in x:
        comparison.append(x[key] == w[key])
    print("Original vs even faster:", np.all(comparison))

    print("Measuring times: \n")
    for f_name in ["original", "faster", "pureNumpy", "evenFaster"]:
        stmt = f_name + '(example_image)'
        print(f_name, timeit.timeit(stmt=stmt, number=1, globals=globals()))
