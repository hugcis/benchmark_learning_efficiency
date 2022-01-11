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
        if mask is not None:
            ret_mask = [[self.enc_size * c + i
                         for i in range(self.enc_size)
                         for c in m]
                        for m in mask]
        else:
            ret_mask = None
        return task, ret_mask

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
                 dictionary: List[str] = ["A", "B", "C", "D", "E"],
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
        del kwargs
        tasks = []
        mask = []
        st = set()
        n_rand = max(max_n_seq // len(self.lengths), 1)
        for t in self.lengths:
            for _ in range(5 * n_rand):
                current_task_mask = []
                left = np.random.choice(
                    self.base_dic + [self.separator_symbol] * 2 * len(self.base_dic),
                    size=int(2.5 * t), replace=True).tolist()
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
                    if ct[tc] >= 10:
                        continue
                    left = left + list(tc) + [self.separator_symbol, str(ct[tc])]
                    current_task_mask.append(len(left) - 1)
                    if np.random.random() > 0.3:
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


def add_no(verb: str, no_names: Optional[list[str]] = None) -> list[str]:
    """ Helper function for the QA tasks. """
    if no_names is not None:
        return ["I", "DO", "NOT", verb] + no_names
    else:
        return []


def add_yes(verb, yes_names: Optional[list[str]] = None) -> list[str]:
    """ Helper function for the QA tasks. """
    if yes_names is not None:
        return ["I", verb] + yes_names
    else:
        return []


def make_sentence(verb: str, yes_names: Optional[list[str]] = None,
                  no_names: Optional[list[str]] = None,
                  link_words: list[str] =["AND", "BUT"]) -> list[str]:
    base = []
    if np.random.random() < 0.5:
        base += add_no(verb, no_names)
        add = add_yes(verb, yes_names)
        if base and add:
            base += [np.random.choice(link_words)]
            base += add
        elif add:
            base += add
    else:
        base += add_yes(verb, yes_names)
        add = add_no(verb, no_names)
        if base and add:
            base += [np.random.choice(link_words)]
            base += add
        elif add:
            base += add
    return base


class ElementaryLanguage(TokenTask):
    def __init__(self,
                 object_names: List[str] = ["PETER", "JOHN", "TOM",
                                            "JAMES", "PAUL"],
                 separator_symbol: str = " ",
                 sentence_term_symbol: str = ".",
                 verbs: List[str] = ["SEE", "HEAR"],
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
        del kwargs
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
            task = make_sentence(verb, yes_names, no_names)
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


class HarderElementaryLanguage(ElementaryLanguage):
    def __init__(self,
                 object_names: List[str] = ["PETER", "JOHN", "TOM",
                                            "JAMES", "PAUL", "MARC",
                                            "LUKE", "SIMON", "ANDREW",
                                            "BRUNO", "LISA"],
                 verbs: List[str] = ["SEE", "HEAR", "CALL", "FEEL", "SMELL"]):
        super().__init__(object_names=object_names, verbs=verbs)
        self.name = "hard-qa"


class ElementaryLanguageWithWorldDef(ElementaryLanguage):
    def __init__(self,
                 object_names: List[str] = ["PETER", "JOHN", "TOM",
                                            "JAMES", "PAUL", "MARC",
                                            "LUKE", "SIMON", "ANDREW",
                                            "BRUNO", "LISA", "HENRI", "LEO"],
                 verbs: List[str] = ["SEE", "HEAR", "CALL", "FEEL", "SMELL",
                                     "UNDERSTAND", "TOUCH"]):
        super().__init__(object_names=object_names, verbs=verbs)
        self.name = "qa-world-def"

    def generate_tasks(self, max_n_seq: int = 10 , **kwargs):
        del kwargs
        tasks: TaskType = []
        mask: Mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset = np.random.choice(self.object_names, size=subset_size,
                                      replace=False)

            # Choose the number of verbs to use
            n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
            verbs = np.random.choice(self.verbs, size=n_verbs, replace=False)

            name_map: Dict[str, list[str]] = {}
            indices = np.random.choice(range(len(subset)), size=n_verbs,
                                       replace=False)
            indices = np.sort(indices)
            for i, verb in enumerate(verbs):
                right = len(subset) if i >= len(verbs) - 1 else indices[i+1]
                name_map[verb] = subset[indices[i]:right].tolist()

            task = []
            yes_map: Dict[str, list[str]] = {}
            no_map: Dict[str, list[str]] = {}
            first = True
            for verb in verbs:
                if not first:
                    task += [self.sentence_term_symbol]
                else:
                    first = False
                # Decide the yes names and the no names (may be empty)
                yes = np.random.randint(len(name_map[verb]) + 1)
                yes_names = np.random.choice(name_map[verb], size=yes,
                                             replace=False).tolist()
                yes_map[verb] = yes_names
                if yes_names:
                    yes_names = " AND ".join(yes_names).split(" ")
                no_names = [i for i in name_map[verb] if i not in yes_names]
                no_map[verb] = no_names
                if no_names:
                    no_names = " AND ".join(no_names).split(" ")

                # Build the sentence
                task += make_sentence(verb, yes_names, no_names)

            # Choose which verb/name we will ask about
            question_verb = str(np.random.choice(verbs))
            tgt = np.random.choice(name_map[question_verb])
            # Make the answer
            task += ([self.sentence_term_symbol] +
                     ["DO", "I", question_verb, tgt, self.query_symbol] +
                     ["YES" if tgt in yes_map[question_verb] else "NO"])
            current_mask.append(len(task) - 1)
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


def make_adj_objs(size_adj: list[str], color_adj: list[str],
                  obj_names: list[str]) -> list[list[str]]:
    """
    Constructs triples of size adjective, color adjective, object name

    Returns:
        A list of lists of objects randomly prefixed.
    """
    output = [[i] for i in np.random.permutation(obj_names)]
    for item in output:
        if np.random.random() > .4:
            item.insert(0, np.random.choice(color_adj))
        if np.random.random() > .4:
            item.insert(0, np.random.choice(size_adj))
    return output

def make_prefix(name):
    if name[0] in ["A", "I", "U", "E", "O"]:
        return "AN"
    else:
        return "A"

class AdjectiveLanguage(TokenTask):
    def __init__(self,
                 object_names: List[str] = ["BANANA", "APPLE", "PEAR",
                                            "PEACH", "APRICOT",
                                            "CAR", "PLANE", "TRAIN"],
                 separator_symbol: str = " ",
                 sentence_term_symbol: str = ".",
                 verbs: List[str] = ["SEE", "HEAR", "CALL", "FEEL", "SMELL", "TOUCH"],
                 color_adj: List[str] = ["RED", "GREEN", "BLUE", "YELLOW"],
                 size_adj: List[str] = ["SMALL", "BIG", "HUGE", "TINY"],
                 query_symbol: str = "?"):
        self.object_names = object_names
        self.color_adj = color_adj
        self.size_adj = size_adj
        self.verbs = verbs
        self.sentence_term_symbol = sentence_term_symbol
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        dictionary = object_names + verbs + color_adj + size_adj
        dictionary += ["I", "DO", "NOT", "AND", "BUT", "WHAT",
                       "A", "AN", "COLOR", "SIZE", "IS", "THE",
                       query_symbol, sentence_term_symbol,
                       "YES", "NO"]
        super().__init__("adj-qa", 0, dictionary)

    def generate_tasks(self, max_n_seq: int = 10 , **kwargs):
        del kwargs
        tasks: TaskType = []
        mask: Mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset: list[str] = np.random.choice(self.object_names, size=subset_size,
                                                 replace=False).tolist()
            adj_subset = make_adj_objs(self.size_adj, self.color_adj, subset)

            # Choose the number of verbs to use
            n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
            verbs: list[str] = np.random.choice(self.verbs, size=n_verbs, replace=False).tolist()

            name_map: Dict[str, list[list[str]]] = {}
            indices = np.random.choice(range(len(subset)), size=n_verbs,
                                       replace=False)
            indices = np.sort(indices)
            for i, verb in enumerate(verbs):
                right = len(subset) if i >= len(verbs) - 1 else indices[i+1]
                name_map[verb] = adj_subset[indices[i]:right]

            task = []
            yes_map: Dict[str, list[list[str]]] = {}
            no_map: Dict[str, list[list[str]]] = {}
            first = True
            for verb in verbs:
                yes_names, no_names = None, None
                if not first:
                    task += [self.sentence_term_symbol]
                else:
                    first = False
                # Decide the yes names and the no names (may be empty)
                yes = np.random.randint(len(name_map[verb]) + 1)
                pre_yes_names: list[list[str]] = [
                    name_map[verb][g]
                    for g in np.random.choice(range(len(name_map[verb])), size=yes, replace=False)
                ]
                yes_map[verb] = pre_yes_names
                if pre_yes_names:
                    flatten_yes_names = [make_prefix(prefixed_name[0]) + self.separator_symbol +
                                         " ".join(prefixed_name)
                                         for prefixed_name in pre_yes_names]
                    yes_names = " AND ".join(flatten_yes_names).split(" ")
                pre_no_names = [i for i in name_map[verb]
                                if i[0] not in [n[0] for n in pre_yes_names]]
                no_map[verb] = pre_no_names
                if pre_no_names:
                    flatten_no_names = [make_prefix(prefixed_name[0]) + self.separator_symbol +
                                        " ".join(prefixed_name)
                                        for prefixed_name in pre_no_names]
                    no_names = " AND ".join(flatten_no_names).split(" ")

                # Build the sentence
                task += make_sentence(verb, yes_names, no_names)

            # Add the question part
            task += self.construct_question(name_map, yes_map, verbs)

            # Last symbol is the one to predict
            current_mask.append(len(task) - 1)

            # Check unicity
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)

    def construct_question(self, name_map: dict[str, list[list[str]]],
                           yes_map: dict[str, list[list[str]]],
                           verbs: list[str]) -> list[str]:
        # Choose which verb/name we will ask about
        candidates = [(k, i) for k, c in yes_map.items() for i in c if len(i) > 1]

        # If no possible answer has any adjective, we force the question to be YES/NO
        if candidates:
            question_chooser = np.random.random()
        else:
            question_chooser = 0

        if question_chooser < 1 / (len(self.color_adj) + len(self.size_adj) + 2):
            # YES/NO
            question_verb = str(np.random.choice(verbs))
            tgt: list[str] = name_map[question_verb][np.random.randint(len(name_map[question_verb]))]
            # Make the answer
            return ([self.sentence_term_symbol] +
                    ["DO", "I", question_verb, make_prefix(tgt[0])] + tgt +
                    [self.query_symbol] +
                    ["YES" if tgt in yes_map[question_verb] else "NO"])
        else:
            # Question about size or color
            question_verb, tgt = candidates[np.random.randint(len(candidates))]
            # Select the adjective we are asking about
            selected_adj: str = np.random.choice(tgt[:-1])
            if selected_adj in self.color_adj:
                question = ["WHAT", "COLOR", "IS", "THE"]
            elif selected_adj in self.size_adj:
                question = ["WHAT", "SIZE", "IS", "THE"]
            else:
                raise ValueError(f"The adjective {selected_adj} is not in the list.")
            # Make the answer
            return ([self.sentence_term_symbol] +
                    question + [tgt[-1], "I", question_verb, self.query_symbol] +
                    [selected_adj])


def print_with_sep(tasks, sep="", lim=10):
    """ Pretty print some examples from a task's generated sequences. """
    print("\n".join([sep.join([str(k) for k in s]) for s in tasks[:lim]]))
