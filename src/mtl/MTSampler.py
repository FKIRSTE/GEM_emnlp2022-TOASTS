import torch.utils.data as data_utils
import math
from datasets import load_dataset, concatenate_datasets

class Sampler:
    def __init__(self, family_dict, size_dict, limit=-1, temperature=4):
        family_size_list = {}
        for task, fam in family_dict.items():
            if fam not in family_size_list.keys():
                family_size_list[fam] = []
            task_size_tuple = (task, size_dict[task])
            family_size_list[fam].append(task_size_tuple)

        self.family_size = family_size_list  # list of all task sizes per family
        self.family_total_size = {f : sum(n for _, n in sizes) for f, sizes in self.family_size.items()}  # :-1 working here? cause tuple...
        self.family = family_dict
        self.size = size_dict
        self.limit = limit
        self.result_size = {task : -1 for task, _ in family_dict.items()}
        self.T = temperature

#region Decorator
    def whole_family(func):
        def wrapper_whole_family(*args):
            for key, _ in args[0].family_total_size.items():
                func(*args, key)
            return args[0].result_size
        return wrapper_whole_family
#endregion
#region Concat

    def __concat(self, task):
        return concatenate_datasets(task)
#endregion
#region WorksOnConcat

    def __limit(self, family, percentage = -1):
        """
        returns the required ds percentage to match the limit. can be >100%
        """
        percentage_set = (percentage > 0.0) & (percentage < 1.0)
        total = self.family_total_size[family]
        for (task, size) in self.family_size[family]:
            if not percentage_set:
                percentage = size/total

            n = math.ceil(percentage*self.limit)
            self.result_size[task] = n
        #return work_fam
#endregion
#region WillConcat
    @whole_family
    def proportional(cls, family):
        """
        keep original length,
        concat,
        """

        if cls.limit > 0:
            cls.__limit(family)

    @whole_family
    def uniform(cls, family):
        """
        find the shortest
        """
        percentage = 1 / len(cls.family_size[family])
        cls.__limit(family, percentage=percentage)

    @whole_family
    def temperature(cls, family):
        total = cls.family_total_size[family]
        percentages = [i/total for _, i in cls.family_size[family]]
        t_scaled_perc = [x**(1/cls.T) for x in percentages]
        norm_perc = [float(i)/sum(t_scaled_perc) for i in t_scaled_perc]
        print("-.----------------------------------")
        print(f"{norm_perc} with Temperature {cls.T}")
        for idx, (task, size) in enumerate(cls.family_size[family]):
            percentage = norm_perc[idx]
            n = math.ceil(percentage*cls.limit)
            cls.result_size[task] = n

#endregion
#region CustomLoad
    def load(self, task, name, addition=None, revision=None, train="train"):
        ds_collect = []
        #print(self.result_size)
        size = self.result_size[task]
        ds_size = self.size[task]
        print(f"{ds_size} vs {size}" )
        while size > ds_size:  # 100:
            ds_collect.append(load_dataset(name, addition, revision=revision, split=f'{train}[:100%]'))
            size = size - ds_size
        ds_collect.append(load_dataset(name, addition, revision=revision, split=f"{train}[:{size}]"))
        return self.__concat(task=ds_collect)
#endregion