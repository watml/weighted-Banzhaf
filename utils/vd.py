import os
import itertools
import fcntl
from tqdm import tqdm
import platform
from datetime import  datetime
import multiprocessing


## this lock tends to loose when using multiprocessing.Pool after locking
class fcntl_lock:
    def __init__(self, path):
        self.file = open(os.path.join(path, ".locker"), "w+")

    def acquire(self):
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            return False
        else:
            return True

    def realse(self):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


class os_lock:
    def __init__(self, path):
        self.lockfile = os.path.join(path, "lockfile")

    def acquire(self):
        try:
            fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL)
        except OSError:
            return False
        else:
            os.close(fd)
            return True

    def realse(self):
        os.remove(self.lockfile)


class test_lock:
    def __init__(self, path):
        self.file = open(os.path.join(path, ".locker"), "w+")
        self.lockfile = os.path.join(path, "lockfile")

    def acquire(self):
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            return False
        else:
            try:
                fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL)
            except OSError:
                with open("/home/wangy1g/wd/datavaluation/lock_err.txt", "a+") as f:
                    f.write("fcntl lock failed!\n")
                return False
            else:
                os.close(fd)
                return True

    def realse(self):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()
        os.remove(self.lockfile)


class vd_tqdm(tqdm):
    def __init__(self, *args, file_to_write, write_mode="a+", **kwargs):
        tqdm.__init__(self, *args, **kwargs)
        self.write_mode = write_mode
        self.file_to_write = file_to_write

        path_chips = file_to_write.split(os.sep)
        self.path = os.sep.join(path_chips[:-1])

    def update(self, n=1):
        displayed = super(vd_tqdm, self).update(n)
        if displayed:
            rate = self.format_dict["rate"]
            if rate:
                remaining = (self.format_dict["total"] - self.format_dict["n"]) / rate
                lock = fcntl_lock(self.path)
                while True:
                    is_lock = lock.acquire()
                    if is_lock:
                        break
                with open(self.file_to_write, self.write_mode) as f:
                    f.write(f"the remaining time on node {platform.node()} " + \
                            f"is {self.format_interval(remaining)} {datetime.now()}\n")
                lock.realse()
        return displayed


def vd_Pool(n=None):
    import os
    # below is the complete list of environmental variables and the package
    # that uses that variable to control the number of threads it spawns.
    os.environ["OMP_NUM_THREADS"] = "1"  # openmp, export OMP_NUM_THREADS=1
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # openblas, export OPENBLAS_NUM_THREADS=1
    os.environ["MKL_NUM_THREADS"] = "1"  # mkl, export MKL_NUM_THREADS=1
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # accelerate, export VECLIB_MAXIMUM_THREADS=1
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # numexpr, export NUMEXPR_NUM_THREADS=1
    return multiprocessing.Pool(n)


class path_parser:
    def __init__(self, path):
        path_split = path.split(";")
        self.dict_var = dict()
        for pair in path_split:
            pair_split = pair.split("=")
            self.dict_var.update({pair_split[0] : pair_split[1]})

    def __call__(self, key):
        return self.dict_var[key]



