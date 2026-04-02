from collections import defaultdict, deque
import datetime
import pickle
import time
import errno
import os

import torch
import torch.distributed as dist


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)  # deque简单理解成加强版list
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):  # @property 是装饰器，这里可简单理解为增加median属性(只读)
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    收集各个进程中的数据
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 进程数
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:  # 单GPU的情况
        return input_dict
    with torch.no_grad():  # 多GPU的情况
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}
        return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        # 初始化函数
        self.meters = defaultdict(SmoothedValue)  # 使用defaultdict来存储不同指标的SmoothedValue对象
        self.delimiter = delimiter  # 分隔符，用于在打印日志时分隔不同的指标

    def update(self, **kwargs):
        # 更新指标值的方法
        for k, v in kwargs.items():  # 遍历传入的指标名和指标值
            if isinstance(v, torch.Tensor):
                v = v.item()  # 如果指标值是Tensor类型，则转换为Python数值
            assert isinstance(v, (float, int))  # 确保指标值是float或int类型
            self.meters[k].update(v)  # 更新对应指标的SmoothedValue对象

    def __getattr__(self, attr):
        # 重写__getattr__方法，用于访问指标值
        if attr in self.meters:
            return self.meters[attr]  # 如果请求的属性在meters中，则返回对应的SmoothedValue对象
        if attr in self.__dict__:
            return self.__dict__[attr]  # 如果请求的属性是对象的直接属性，则直接返回
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))  # 如果都不是，则抛出属性错误

    def __str__(self):
        # 重写__str__方法，定义对象的字符串表示
        loss_str = []
        for name, meter in self.meters.items():  # 遍历所有指标
            loss_str.append("{}: {}".format(name, str(meter)))  # 将指标名和指标值格式化为字符串
        return self.delimiter.join(loss_str)  # 使用分隔符连接所有指标字符串

    def synchronize_between_processes(self):
        # 在多进程环境下同步指标值的方法
        for meter in self.meters.values():
            meter.synchronize_between_processes()  # 调用每个SmoothedValue对象的同步方法

    def add_meter(self, name, meter):
        # 添加新的指标
        self.meters[name] = meter  # 在meters字典中添加新的指标名和SmoothedValue对象

    def log_every(self, iterable, print_freq, header=None):
        # 日志记录函数，定期打印指标值
        i = 0
        if not header:
            header = ""  # 如果没有提供头部信息，则设置为空字符串
        start_time = time.time()  # 记录开始时间
        end = time.time()  # 初始化结束时间
        iter_time = SmoothedValue(fmt='{avg:.4f}')  # 迭代时间的SmoothedValue对象
        data_time = SmoothedValue(fmt='{avg:.4f}')  # 数据加载时间的SmoothedValue对象
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"  # 格式化字符串，用于打印进度
        # 根据是否有CUDA可用，选择不同的日志消息格式
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}',
                                           'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join([header,
                                           '[{0' + space_fmt + '}/{1}]',
                                           'eta: {eta}',
                                           '{meters}',
                                           'time: {time}',
                                           'data: {data}'])
        MB = 1024.0 * 1024.0  # 定义MB的大小，用于内存计算
        for obj in iterable:  # 遍历可迭代对象
            data_time.update(time.time() - end)  # 更新数据加载时间
            yield obj  # 产生当前对象
            iter_time.update(time.time() - end)  # 更新迭代时间
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 如果达到打印频率或迭代结束，则打印日志信息
                eta_second = int(iter_time.global_avg * (len(iterable) - i))  # 估计剩余时间
                eta_string = str(datetime.timedelta(seconds=eta_second))  # 将剩余时间格式化为字符串
                if torch.cuda.is_available():
                    # 如果有CUDA可用，则打印包含最大内存使用量的日志信息
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time),
                                         memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    # 否则，打印不包含内存使用量的日志信息
                    print(log_msg.format(i, len(iterable),
                                         eta=eta_string,
                                         meters=str(self),
                                         time=str(iter_time),
                                         data=str(data_time)))
            i += 1  # 更新迭代计数
            end = time.time()  # 更新结束时间
        total_time = time.time() - start_time  # 计算总时间
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))  # 将总时间格式化为字符串
        print('{} Total time: {} ({:.4f} s / it)'.format(header,
                                                         total_time_str,
                                                         total_time / len(iterable)))  # 打印总时间和平均迭代时间


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        """根据step数返回一个学习率倍率因子"""
        if x >= warmup_iters:  # 当迭代数大于给定的warmup_iters时，倍率因子为1
            return 1
        alpha = float(x) / warmup_iters
        # 迭代过程中倍率因子从warmup_factor -> 1
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """检查是否支持分布式环境"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

