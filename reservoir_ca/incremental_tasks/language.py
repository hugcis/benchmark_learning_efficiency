from typing import List, Optional, Dict

import numpy as np

from .tasks import TokenTask, TaskType, choose_minimal_set


def add_no(verb: str, no_names: Optional[list[str]] = None) -> list[str]:
    """Helper function for the QA tasks."""
    if no_names is not None and no_names:
        return ["I", "DO", "NOT", verb] + no_names
    else:
        return []


def add_yes(verb, yes_names: Optional[list[str]] = None) -> list[str]:
    """Helper function for the QA tasks."""
    if yes_names is not None and yes_names:
        return ["I", verb] + yes_names
    else:
        return []


def make_sentence(
    verb: str,
    yes_names: Optional[list[str]] = None,
    no_names: Optional[list[str]] = None,
    link_words: list[str] = ["AND", "BUT"],
) -> list[str]:
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
    def __init__(
        self,
        object_names: List[str] = ["PETER", "JOHN", "TOM", "JAMES", "PAUL"],
        separator_symbol: str = " ",
        sentence_term_symbol: str = ".",
        verbs: List[str] = ["SEE", "HEAR"],
        query_symbol: str = "?",
    ):
        self.object_names = object_names
        self.verbs = verbs
        self.sentence_term_symbol = sentence_term_symbol
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        dictionary = object_names + verbs  # + color_adj
        dictionary += [
            "I",
            "DO",
            "NOT",
            "AND",
            "BUT",
            query_symbol,
            sentence_term_symbol,
            "YES",
            "NO",
        ]
        super().__init__("qa", 0, dictionary)

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        del kwargs
        tasks: TaskType = []
        mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []
            verb = np.random.choice(self.verbs)

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset = np.random.choice(
                self.object_names, size=subset_size, replace=False
            )

            # Decide the yes names and the no names (may be empty)
            yes = np.random.randint(len(subset))
            yes_names = np.random.choice(subset, size=yes, replace=False).tolist()
            if yes_names:
                yes_names = " AND ".join(yes_names).split(" ")
            no_names = [i for i in subset if i not in yes_names]
            if no_names:
                no_names = " AND ".join(no_names).split(" ")

            # Build the sentence
            task = make_sentence(verb, yes_names, no_names)
            tgt = np.random.choice(subset)
            # Make the answer
            task += (
                [self.sentence_term_symbol]
                + ["DO", "I", verb, tgt, self.query_symbol]
                + ["YES" if tgt in yes_names else "NO"]
            )
            current_mask.append(len(task) - 1)
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


class HarderElementaryLanguage(ElementaryLanguage):
    def __init__(
        self,
        object_names: List[str] = [
            "PETER",
            "JOHN",
            "TOM",
            "JAMES",
            "PAUL",
            "MARC",
            "LUKE",
            "SIMON",
            "ANDREW",
            "BRUNO",
            "LISA",
        ],
        verbs: List[str] = ["SEE", "HEAR", "CALL", "FEEL", "SMELL"],
    ):
        super().__init__(object_names=object_names, verbs=verbs)
        self.name = "hard-qa"


class ElementaryLanguageWithWorldDef(ElementaryLanguage):
    def __init__(
        self,
        object_names: List[str] = [
            "PETER",
            "JOHN",
            "TOM",
            "JAMES",
            "PAUL",
            "MARC",
            "LUKE",
            "SIMON",
            "ANDREW",
            "BRUNO",
            "LISA",
            "HENRI",
            "LEO",
        ],
        verbs: List[str] = [
            "SEE",
            "HEAR",
            "CALL",
            "FEEL",
            "SMELL",
            "UNDERSTAND",
            "TOUCH",
        ],
    ):
        super().__init__(object_names=object_names, verbs=verbs)
        self.name = "qa-world-def"

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        del kwargs
        tasks: TaskType = []
        mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset = np.random.choice(
                self.object_names, size=subset_size, replace=False
            )

            # Choose the number of verbs to use
            n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
            verbs = np.random.choice(self.verbs, size=n_verbs, replace=False)

            name_map: Dict[str, list[str]] = {}
            indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
            indices = np.sort(indices)
            for i, verb in enumerate(verbs):
                right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
                name_map[verb] = subset[indices[i] : right].tolist()

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
                yes_names = np.random.choice(
                    name_map[verb], size=yes, replace=False
                ).tolist()
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
            task += (
                [self.sentence_term_symbol]
                + ["DO", "I", question_verb, tgt, self.query_symbol]
                + ["YES" if tgt in yes_map[question_verb] else "NO"]
            )
            current_mask.append(len(task) - 1)
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


class ElementaryLanguageWithWorldDefCounting(ElementaryLanguageWithWorldDef):
    def __init__(
        self,
        object_names: List[str] = [
            "PETER",
            "JOHN",
            "TOM",
            "JAMES",
            "PAUL",
            "MARC",
            "LUKE",
            "SIMON",
            "ANDREW",
            "BRUNO",
            "LISA",
            "HENRI",
            "LEO",
        ],
        verbs: List[str] = [
            "SEE",
            "HEAR",
            "CALL",
            "FEEL",
            "SMELL",
            "UNDERSTAND",
            "TOUCH",
        ],
        numbers: List[str] = [
            "ZERO",
            "ONE",
            "TWO",
            "THREE",
            "FOUR",
            "FIVE",
            "SIX",
            "SEVEN",
            "EIGHT",
            "NINE",
            "TEN",
            "ELEVEN",
            "TWELVE",
        ],
    ):
        super().__init__(object_names=object_names, verbs=verbs)
        self.name = "qa-world-def-ct"
        self.number_map = dict(enumerate(numbers))
        self.dictionary += numbers
        self.dictionary += ["HOW", "MANY", "PEOPLE"]

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        del kwargs
        tasks: TaskType = []
        mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset = np.random.choice(
                self.object_names, size=subset_size, replace=False
            )

            # Choose the number of verbs to use
            n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
            verbs = np.random.choice(self.verbs, size=n_verbs, replace=False)

            name_map: Dict[str, list[str]] = {}
            indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
            indices = np.sort(indices)
            for i, verb in enumerate(verbs):
                right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
                name_map[verb] = subset[indices[i] : right].tolist()

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
                yes_names = np.random.choice(
                    name_map[verb], size=yes, replace=False
                ).tolist()
                yes_map[verb] = yes_names
                if yes_names:
                    yes_names = " AND ".join(yes_names).split(" ")
                no_names = [i for i in name_map[verb] if i not in yes_names]
                no_map[verb] = no_names
                if no_names:
                    no_names = " AND ".join(no_names).split(" ")

                # Build the sentence
                task += make_sentence(verb, yes_names, no_names)

            coin_up = np.random.random() > 0.5
            if coin_up:
                # Choose which verb/name we will ask about
                question_verb = str(np.random.choice(verbs))
                tgt = len(yes_map[question_verb])
                # Make the answer
                task += (
                    [self.sentence_term_symbol]
                    + [
                        "HOw",
                        "MANY",
                        "PEOPLE",
                        "DO",
                        "I",
                        question_verb,
                        self.query_symbol,
                    ]
                    + [self.number_map[tgt]]
                )

            else:
                # Choose which verb/name we will ask about
                question_verb = str(np.random.choice(verbs))
                tgt = np.random.choice(name_map[question_verb])
                # Make the answer
                task += (
                    [self.sentence_term_symbol]
                    + ["DO", "I", question_verb, tgt, self.query_symbol]
                    + ["YES" if tgt in yes_map[question_verb] else "NO"]
                )
            current_mask.append(len(task) - 1)
            task_str = self.separator_symbol.join(task)
            if task_str not in st:
                tasks.append(task)
                mask.append(current_mask)
                st.add(task_str)
        return choose_minimal_set(tasks, max_n_seq, mask=mask)


def make_adj_objs(
    size_adj: list[str], color_adj: list[str], obj_names: list[str]
) -> list[list[str]]:
    """
    Constructs triples of size adjective, color adjective, object name

    Returns:
        A list of lists of objects randomly prefixed.
    """
    output = [[i] for i in np.random.permutation(obj_names)]
    for item in output:
        if np.random.random() > 0.4:
            item.insert(0, np.random.choice(color_adj))
        if np.random.random() > 0.4:
            item.insert(0, np.random.choice(size_adj))
    return output


def make_prefix(name):
    if name[0] in ["A", "I", "U", "E", "O"]:
        return "AN"
    else:
        return "A"


DEFAULT_OBJECT_NAMES = [
    "BANANA",
    "APPLE",
    "PEAR",
    "PEACH",
    "APRICOT",
    "CAR",
    "PLANE",
    "TRAIN",
]


class AdjectiveLanguage(TokenTask):
    def __init__(
        self,
        object_names: List[str] = DEFAULT_OBJECT_NAMES,
        separator_symbol: str = " ",
        sentence_term_symbol: str = ".",
        verbs: List[str] = ["SEE", "HEAR", "CALL", "FEEL", "SMELL", "TOUCH"],
        color_adj: List[str] = ["RED", "GREEN", "BLUE", "YELLOW"],
        size_adj: List[str] = ["SMALL", "BIG", "HUGE", "TINY"],
        query_symbol: str = "?",
    ):
        self.object_names = object_names
        self.color_adj = color_adj
        self.size_adj = size_adj
        self.verbs = verbs
        self.sentence_term_symbol = sentence_term_symbol
        self.query_symbol = query_symbol
        self.separator_symbol = separator_symbol
        dictionary = object_names + verbs + color_adj + size_adj
        dictionary += ["I", "DO", "NOT", "AND", "BUT", "WHAT", "A", "AN"]
        dictionary += ["COLOR", "SIZE", "IS", "THE"]
        dictionary += [query_symbol, sentence_term_symbol, "YES", "NO"]
        super().__init__("adj-qa", 0, dictionary)

    def generate_tasks(self, max_n_seq: int = 10, **kwargs):
        del kwargs
        tasks: TaskType = []
        mask = []
        st = set()
        for _ in range(max_n_seq):
            current_mask = []

            # Choose a subset of object names to work with
            subset_size = np.random.randint(1, len(self.object_names))
            subset: list[str] = np.random.choice(
                self.object_names, size=subset_size, replace=False
            ).tolist()
            adj_subset = make_adj_objs(self.size_adj, self.color_adj, subset)

            # Choose the number of verbs to use
            n_verbs = np.random.randint(1, min(len(self.verbs), subset_size) + 1)
            verbs: list[str] = np.random.choice(
                self.verbs, size=n_verbs, replace=False
            ).tolist()

            name_map: Dict[str, list[list[str]]] = {}
            indices = np.random.choice(range(len(subset)), size=n_verbs, replace=False)
            indices = np.sort(indices)
            for i, verb in enumerate(verbs):
                right = len(subset) if i >= len(verbs) - 1 else indices[i + 1]
                name_map[verb] = adj_subset[indices[i] : right]

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
                    for g in np.random.choice(
                        range(len(name_map[verb])), size=yes, replace=False
                    )
                ]
                yes_map[verb] = pre_yes_names
                if pre_yes_names:
                    flatten_yes_names = [
                        make_prefix(prefixed_name[0])
                        + self.separator_symbol
                        + " ".join(prefixed_name)
                        for prefixed_name in pre_yes_names
                    ]
                    yes_names = " AND ".join(flatten_yes_names).split(" ")
                pre_no_names = [
                    i
                    for i in name_map[verb]
                    if i[0] not in [n[0] for n in pre_yes_names]
                ]
                no_map[verb] = pre_no_names
                if pre_no_names:
                    flatten_no_names = [
                        make_prefix(prefixed_name[0])
                        + self.separator_symbol
                        + " ".join(prefixed_name)
                        for prefixed_name in pre_no_names
                    ]
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

    def construct_question(
        self,
        name_map: dict[str, list[list[str]]],
        yes_map: dict[str, list[list[str]]],
        verbs: list[str],
    ) -> list[str]:
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
            tgt: list[str] = name_map[question_verb][
                np.random.randint(len(name_map[question_verb]))
            ]
            # Make the answer
            return (
                [self.sentence_term_symbol]
                + ["DO", "I", question_verb, make_prefix(tgt[0])]
                + tgt
                + [self.query_symbol]
                + ["YES" if tgt in yes_map[question_verb] else "NO"]
            )
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
            return (
                [self.sentence_term_symbol]
                + question
                + [tgt[-1], "I", question_verb, self.query_symbol]
                + [selected_adj]
            )
