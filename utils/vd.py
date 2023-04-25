import os
import itertools
import fcntl
from tqdm import tqdm
import platform
from datetime import  datetime

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
        super(vd_tqdm, self).__init__(*args, **kwargs)
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








def dict_product(d):
    keys = d.keys()
    vals = d.values()
    args_all = []
    for instance in itertools.product(*vals):
        args_all.append(dict(zip(keys, instance)))
    return args_all

