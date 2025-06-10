# backward_chaining_fol.py

from collections import defaultdict
from typing import List, Union

class AND:
    def __init__(self, *args):
        self.args = list(args)
    def __repr__(self):
        return f"AND({', '.join(map(str, self.args))})"

class OR:
    def __init__(self, *args):
        self.args = list(args)
    def __repr__(self):
        return f"OR({', '.join(map(str, self.args))})"

class Rule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent  
        self.consequent = consequent  

    def __repr__(self):
        return f"IF {self.antecedent} THEN {self.consequent}"

def match(statement: str, hypothesis: str):
    if statement == hypothesis:
        return {}
    s_parts = statement.split()
    h_parts = hypothesis.split()
    if len(s_parts) != len(h_parts):
        return None
    bindings = {}
    for s, h in zip(s_parts, h_parts):
        if s.startswith("(?") and s.endswith(")"):
            bindings[s] = h
        elif s != h:
            return None
    return bindings

def populate(expr, bindings):
    if isinstance(expr, str):
        for var, val in bindings.items():
            expr = expr.replace(var, val)
        return expr
    elif isinstance(expr, (AND, OR)):
        return type(expr)(*[populate(arg, bindings) for arg in expr.args])
    return expr

def simplify(expr):
    if isinstance(expr, (AND, OR)):
        flat_args = []
        for arg in expr.args:
            arg = simplify(arg)
            if isinstance(arg, type(expr)):
                flat_args.extend(arg.args)
            else:
                flat_args.append(arg)
        seen = set()
        result_args = []
        for arg in flat_args:
            if str(arg) not in seen:
                seen.add(str(arg))
                result_args.append(arg)
        return type(expr)(*result_args)
    return expr

def backchain(rules: List[Rule], hypothesis: str) -> Union[str, AND, OR]:
    goals = [hypothesis]
    for rule in rules:
        bindings = match(rule.consequent, hypothesis)
        if bindings is not None:
            instantiated = populate(rule.antecedent, bindings)
            if isinstance(instantiated, str):
                goals.append(backchain(rules, instantiated))
            elif isinstance(instantiated, (AND, OR)):
                subgoals = [backchain(rules, sub) for sub in instantiated.args]
                goals.append(type(instantiated)(*subgoals))
    return simplify(OR(*goals))

if __name__ == "__main__":
    rules = [
        Rule("(?x) is a bird", "(?x) is an animal"),
        Rule(AND("(?x) has feathers", "(?x) lays eggs"), "(?x) is a bird"),
        Rule("(?x) is a mammal", "(?x) is an animal")
    ]

    hypothesis = "tweety is an animal"
    tree = backchain(rules, hypothesis)
    print("Goal Tree:", tree)
