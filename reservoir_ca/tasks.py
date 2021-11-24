from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Type, Union, Sequence, Dict, List, Tuple, Optional
import collections

TaskType = List[List[str]]
Mask = Optional[List[List[int]]]


def choose_minimal_set(tasks: TaskType, max_n_seq: int,
                       mask: Mask = None) -> Tuple[TaskType, Mask]:
    if len(tasks) > max_n_seq:
        idx = np.random.choice(range(len(tasks)), size=max_n_seq,
                                replace=False)
        if mask is not None:
            return_mask = [mask[i] for i in idx]
        else:
            return_mask = None
        return [tasks[i] for i in idx], return_mask
    else:
        return tasks, mask


class Task(ABC):
    """
    Abstract base class for tasks. A task should have a dictionary member that
    contains all the possible symbols used in the generated sequences
    """
    dictionary: List[str]

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_tasks(self, max_n_seq: int = 10, **kwargs) -> Tuple[TaskType, Mask]:
        pass
        del self, max_n_seq, kwargs

    def output_dimension(self) -> int:
        return len(self.dictionary)

    def set_lengths(self, lengths: Union[int, Sequence[int]]):
        if isinstance(lengths, int) or len(lengths) == 1:
            if not isinstance(lengths, int):
                l = lengths[0]
            else:
                l = lengths
            self.lengths = list(range(1, l + 1))
        elif len(lengths) == 2:
            if lengths[1] > lengths[0]:
                self.lengths = list(range(lengths[0], lengths[1]))
            else:
                raise ValueError("Wrong lengths")
        else:
            self.lengths = lengths


class HybridTask(Task):
    def __init__(self, named_tasks: Dict[str, Type[Task]],
                 task_args: Dict[str, List[Any]]):
        self.named_tasks = {}
        # With this, we create a new instance of each subtask every time we create a new
        # HybridTask
        for n in named_tasks:
            self.named_tasks[n] = named_tasks[n](*task_args.get(n, []))
        super().__init__("hyb_{}".format("_".join(t.name for t in self.named_tasks.values())))

        # The dictionary is the union of all subtask dictionaries
        set_dictionary = set()
        for task in self.named_tasks.values():
            set_dictionary.update(task.dictionary)
        self.dictionary = list(set_dictionary)

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        res: TaskType = []
        msk: Mask = None
        # Each task contributes a fraction of the total sequences
        max_n_per_task = max_n_seq // len(self.named_tasks)
        for n in self.named_tasks:
            task, mask = self.named_tasks[n].generate_tasks(max_n_seq=max_n_per_task, **kwargs)
            res = res + task
            # TODO Take care of cases where some tasks have masks and others don't
            if mask is not None:
                if msk is None:
                    msk = []
                msk = msk + mask
        return res, msk


class BinaryTask(Task):
    def __init__(self, name: str, lengths: Union[int, Sequence[int]]):
        super().__init__(name)
        self.dictionary = ["0", "1"]
        self.set_lengths(lengths)


class TokenTask(Task):
    def __init__(self, name: str, lengths: Union[int, Sequence[int]],
                 dictionary: Sequence[str] = ["A", "B", "C"]):
        super().__init__(name)
        self.dictionary = list(dictionary)
        self.set_lengths(lengths)


class BinarizedTask(Task):
    """ Binarized version of a token class """
    def __init__(self, base_task: TokenTask):
        super().__init__("bin_{}".format(base_task.name))
        self.base_task = base_task
        self.dictionary = ["0", "1"]

        self.enc_size = int(np.ceil(np.log2(len(self.base_task.dictionary))))
        formatter = f"{{:0{self.enc_size}b}}"
        self.mapping = {
            d: formatter.format(n)
            for n, d in enumerate(self.base_task.dictionary)
        }

    def convert_to_binary(self, task: TaskType, mask: Mask) -> Tuple[TaskType, Mask]:
        task = [[c  for g in t for c in self.mapping[g]] for t in task]
        mask = [[self.enc_size * c + i for i in range(self.enc_size) for c in m] for m in mask]
        return task, mask

    def generate_tasks(self, max_n_seq: int, **kwargs) -> Tuple[TaskType, Mask]:
        task, mask = self.base_task.generate_tasks(max_n_seq=max_n_seq, **kwargs)
        return self.convert_to_binary(task, mask)


class Periodic(BinaryTask):
    """ Generate all binary periodic sequences with lengths. """
    def __init__(self, lengths: Union[int, Sequence[int]]):
        super().__init__("periodic", lengths)

    def generate_tasks(self, seq_len: int = 100, max_n_seq: int = 10) -> Tuple[TaskType, Mask]:
        tasks = []
        st = set()
        for t in self.lengths:
            for s in range(2 ** t):
                ft = "{{0:0{}b}}".format(t)
                base = [int(i) for i in ft.format(s)]
                task = base * (seq_len // len(base))
                task = (task + base[:seq_len - len(task)])[:]
                task = [str(i) for i in task]
                task_str = "".join(task)
                if task_str not in st:
                    tasks.append(task)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq)


class IncreasingPeriod(BinaryTask):
    """ Generate all binary periodic sequences with increasing periods with
        lengths.
    """
    def __init__(self, lengths: Union[int, Sequence[int]]):
        super().__init__("inc-per", lengths)

    def generate_tasks(self, seq_len: int = 100, max_n_seq: int = 10) -> Tuple[TaskType, Mask]:
        tasks = []
        st = set()
        for t in self.lengths:
            for s in range(2 ** t):
                ft = "{{0:0{}b}}".format(t)
                base = [int(i) for i in ft.format(s)]
                task = base[:]
                ct = 2
                while len(task) < seq_len:
                    task = task + np.concatenate(
                        [np.array(base)[:, None]
                         for _ in range(ct)], axis=1).reshape(-1).tolist()
                    ct += 1

                task = task[:seq_len]
                task = [str(i) for i in task]
                task_str = "".join(task)
                if task_str not in st:
                    tasks.append(task)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq)


class RandomPeriodic(BinaryTask):
    """ Generate random sequences with N-Grams of length in lenghts. """
    def __init__(self, lengths: Union[int, Sequence[int]]):
        super().__init__("rand-per", lengths)

    def generate_tasks(self, seq_len: int = 100, max_n_seq: int = 10) -> Tuple[TaskType, Mask]:
        tasks = []
        st = set()
        for t in self.lengths:
            for _ in range(max_n_seq):
                seq = np.random.randint(2 ** t, size=1 + seq_len // t)
                ft = "{{0:0{}b}}".format(t)
                task = sum([[int(i) for i in ft.format(q)]
                            for q in seq], [])[:seq_len]

                task = [str(i) for i in task]
                task_str = "".join(task)
                if task_str not in st:
                    tasks.append(task)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq)


class SymbolCounting(TokenTask):
    def __init__(self, lengths: Union[int, Sequence[int]],
                 dictionary: List[str] = ["A", "B", "C"],
                 query_symbol: str = "x"):
        super().__init__("sym-ct", lengths, dictionary +
                         [query_symbol] +
                         [str(i) for i in range(10)])
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.base_dic = dictionary

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        del kwargs
        tasks = []
        mask = []
        st = set()
        n_rand = max(max_n_seq // len(self.lengths), 1)
        for t in self.lengths:
            for _ in range(3 * n_rand):
                current_task_mask = []
                n_queries = np.random.randint(1, len(self.base_dic) + 1)
                left = np.random.choice(self.base_dic, size=t, replace=True).tolist()
                ct = collections.Counter(left)

                # For now avoid the case with more than ten times the same symbol as it
                # makes us choose between treating 10 and plus as separate tokens or
                # as two single digit tokens
                if any(c >= 10 for c in ct.values()):
                    continue

                tk = np.random.choice(self.base_dic, size=n_queries, replace=False)
                for tc in tk:
                    left = left + [self.query_symbol, tc, str(ct[tc])]
                    current_task_mask.append(len(left) - 1)
                task_str = "".join(left)
                if task_str not in st:
                    tasks.append(left[:])
                    mask.append(current_task_mask[:])
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


class HardSymbolCounting(TokenTask):
    def __init__(self, lengths: Union[int, Sequence[int]],
                 dictionary: List[str] = ["A", "B", "C"],
                 separator_symbol: str = "y",
                 query_symbol: str = "x"):
        super().__init__("hard-sym-ct", lengths, dictionary +
                         [query_symbol, separator_symbol] +
                         [str(i) for i in range(10)])
        assert query_symbol not in dictionary
        assert np.all([len(i) == 1 for i in dictionary])
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        self.base_dic = dictionary + [separator_symbol]

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        tasks = []
        mask = []
        st = set()
        n_rand = max(max_n_seq // len(self.lengths), 1)
        for t in self.lengths:
            for _ in range(5 * n_rand):
                current_task_mask = []
                left = np.random.choice(
                    self.base_dic + [self.separator_symbol] * 2 * len(self.base_dic),
                    size=t, replace=True).tolist()
                while left and left[0] == self.separator_symbol:
                    left.pop(0)
                while left and left[-1] == self.separator_symbol:
                    left.pop(0)
                if not left:
                    continue
                # No duplicates separator symbols
                left = [v for i, v in enumerate(left)
                        if (i == 0
                            or (v == self.separator_symbol
                                and v != left[i-1])
                            or v != self.separator_symbol)]

                ct = collections.Counter("".join(left).split(self.separator_symbol))
                n_queries = np.random.randint(1, len(ct.keys()) + 1)

                tk = np.random.choice(list(ct.keys()), size=n_queries,
                                      replace=False)
                left = left + [self.query_symbol]
                for tc in tk:
                    left = left + list(tc) + [self.separator_symbol, str(ct[tc])]
                    current_task_mask.append(len(left) - 1)
                    if np.random.random() > 0.5:
                        negative = list(3 * tc[:])
                        np.random.shuffle(negative)
                        negative = negative[:len(tc) + np.random.randint(-2, 3)]
                        if negative and not "".join(negative) in ct:
                            left = left + negative + [self.separator_symbol, "0"]
                            current_task_mask.append(len(left) - 1)

                task_str = "".join(left)
                if task_str not in st:
                    tasks.append(left[:])
                    mask.append(current_task_mask)
                    st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


def add_no(no_names, verb):
        if no_names:
            return ["I", "DO", "NOT", verb] + no_names
        else:
            return []


def add_yes(yes_names, verb):
    if yes_names:
        return ["I", verb] + yes_names
    else:
        return []


def make_sentence(yes_names, no_names, verb, link_words=["AND", "BUT"]):
    base = []
    if np.random.random() < 0.5:
        base += add_no(no_names, verb)
        add = add_yes(yes_names, verb)
        if base and add:
            base += [np.random.choice(link_words)]
            base += add
        elif add:
            base += add
    else:
        base += add_yes(yes_names, verb)
        add = add_no(no_names, verb)
        if base and add:
            base += [np.random.choice(link_words)]
            base += add
        elif add:
            base += add
    return base


def make_adj_sentences(size_adj, color_adj, obj_names):
    output = [[i] for i in np.random.permutation(obj_names)]
    for item in output:
        if np.random.random() > .4:
            item.insert(0, np.random.choice(color_adj))
        if np.random.random() > .4:
            item.insert(0, np.random.choice(size_adj))
    return output


class ElementaryLanguage(TokenTask):
    def __init__(self,
                 object_names: List[str] = ["PETER", "JOHN", "TOM",
                                                "JAMES", "PAUL"],
                 separator_symbol: str = " ",
                 sentence_term_symbol: str = ".",
                 verbs: List[str] = ["SEE", "HEAR"],
                 #color_adj: Sequence[str] = ["RED", "GREEN", "BLUE"],
                 #size_adj: Sequence[str] = ["SMALL", "BIG"],
                 query_symbol: str = "?"):
        self.object_names = object_names
        self.verbs = verbs
        self.sentence_term_symbol = sentence_term_symbol
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        dictionary = object_names + verbs #+ color_adj
        dictionary += ["I", "DO", "NOT", "AND", "BUT",
                       query_symbol,
                       sentence_term_symbol, "YES", "NO"]
        super().__init__("qa", 0, dictionary)

    def generate_tasks(self, max_n_seq: int = 10 , **kwargs):
        tasks: TaskType = []
        mask: Mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []
            verb = np.random.choice(self.verbs)

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset = np.random.choice(self.object_names, size=subset_size,
                                      replace=False)

            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(subset))
            yes_names = np.random.choice(subset, size=yes,
                                         replace=False).tolist()
            if yes_names:
                yes_names = " AND ".join(yes_names).split(" ")
            no_names = [i for i in subset if i not in yes_names]
            if no_names:
                no_names = " AND ".join(no_names).split(" ")

            # Build the sentence
            task = make_sentence(yes_names, no_names, verb)
            tgt = np.random.choice(subset)
            # Make the answer
            task += ([self.sentence_term_symbol] +
                     ["DO", "I", verb, tgt, self.query_symbol] +
                     ["YES" if tgt in yes_names else "NO"])
            current_mask.append(len(task) - 1)
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)



class AdjectiveLanguage(TokenTask):
    def __init__(self,
                 object_names: List[str] = ["BANANA", "APPLE", "PEAR",
                                            "PEACH", "APRICOT",
                                            "CAR", "PLANE", "TRAIN"],
                 separator_symbol: str = " ",
                 sentence_term_symbol: str = ".",
                 verbs: List[str] = ["SEE", "HEAR"],
                 color_adj: List[str] = ["RED", "GREEN", "BLUE"],
                 size_adj: List[str] = ["SMALL", "BIG"],
                 query_symbol: str = "?"):
        self.object_names = object_names
        self.color_adj = color_adj
        self.size_adj = size_adj
        self.verbs = verbs
        self.sentence_term_symbol = sentence_term_symbol
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        dictionary = object_names + verbs + color_adj + size_adj
        dictionary += ["I", "DO", "NOT", "AND", "BUT",
                       query_symbol,
                       sentence_term_symbol, "YES", "NO"]
        super().__init__("qa", 0, dictionary)

    def generate_tasks(self, max_n_seq: int = 10 , **kwargs):
        tasks: TaskType = []
        mask: Mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []
            verb = np.random.choice(self.verbs)

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset = np.random.choice(self.object_names, size=subset_size,
                                      replace=False)

            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(subset))
            yes_names = np.random.choice(subset, size=yes,
                                         replace=False).tolist()
            if yes_names:
                yes_names = " AND ".join(yes_names).split(" ")
            no_names = [i for i in subset if i not in yes_names]
            if no_names:
                no_names = " AND ".join(no_names).split(" ")

            # Build the sentence
            task = make_sentence(yes_names, no_names, verb)
            tgt = np.random.choice(subset)
            # Make the answer
            task += ([self.sentence_term_symbol] +
                     ["DO", "I", verb, tgt, self.query_symbol] +
                     ["YES" if tgt in yes_names else "NO"])
            current_mask.append(len(task) - 1)
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


def print_with_sep(tasks, sep="", lim=10):
    """ Pretty print some examples from a task's generated sequences. """
    print("\n".join([sep.join([str(k) for k in s]) for s in tasks[:lim]]))
