"""Hindley-Milner core utilities for Theta.

This module provides a small, self-contained implementation of the
core HM primitives used for type inference:

- Type representations: TypeVar, TypeConst, TypeArrow, TypeList
- Scheme (polymorphic type scheme) with generalize/instantiate
- Substitution helpers and `unify` algorithm

The goal is to keep this file focused and easy to test independently
from the rest of the interpreter. It is *not* a complete production
HM engine, but it is sufficient to bootstrap inference and can be
extended in later steps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Optional, List, Tuple
import itertools


class Type:
    def apply(self, subst: Dict['TypeVar', 'Type']) -> 'Type':
        raise NotImplementedError

    def vars(self) -> Set['TypeVar']:
        raise NotImplementedError


@dataclass(frozen=True)
class TypeVar(Type):
    name: str

    def apply(self, subst: Dict['TypeVar', Type]) -> Type:
        return subst.get(self, self)

    def vars(self) -> Set['TypeVar']:
        return {self}

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class TypeConst(Type):
    name: str

    def apply(self, subst: Dict['TypeVar', Type]) -> Type:
        return self

    def vars(self) -> Set['TypeVar']:
        return set()

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class TypeArrow(Type):
    left: Type
    right: Type

    def apply(self, subst: Dict['TypeVar', Type]) -> Type:
        return TypeArrow(self.left.apply(subst), self.right.apply(subst))

    def vars(self) -> Set['TypeVar']:
        return self.left.vars() | self.right.vars()

    def __repr__(self):
        left = repr(self.left)
        right = repr(self.right)
        return f'({left} -> {right})'


@dataclass(frozen=True)
class TypeList(Type):
    elem: Type

    def apply(self, subst: Dict['TypeVar', Type]) -> Type:
        return TypeList(self.elem.apply(subst))

    def vars(self) -> Set['TypeVar']:
        return self.elem.vars()

    def __repr__(self):
        return f'[{repr(self.elem)}]'


@dataclass
class Scheme:
    vars: List[TypeVar]
    type: Type

    def __repr__(self):
        if not self.vars:
            return repr(self.type)
        vs = ','.join(v.name for v in self.vars)
        return f'forall {vs}. {repr(self.type)}'


# Fresh variable generator
_var_counter = itertools.count()


def fresh_type_var(prefix: str = 'a') -> TypeVar:
    return TypeVar(f'{prefix}{next(_var_counter)}')


def type_vars_of_type(t: Type) -> Set[TypeVar]:
    return t.vars()


def type_vars_of_scheme(s: Scheme) -> Set[TypeVar]:
    return type_vars_of_type(s.type) - set(s.vars)


def apply_subst(subst: Dict[TypeVar, Type], t: Type) -> Type:
    return t.apply(subst)


def compose_subst(s1: Dict[TypeVar, Type], s2: Dict[TypeVar, Type]) -> Dict[TypeVar, Type]:
    # apply s1 to all types in s2 then merge
    result = {tv: apply_subst(s1, ty) for tv, ty in s2.items()}
    result.update(s1)
    return result


class UnifyError(Exception):
    pass


def occurs_check(var: TypeVar, t: Type, subst: Dict[TypeVar, Type]) -> bool:
    t2 = apply_subst(subst, t)
    return var in t2.vars()


def unify(t1: Type, t2: Type, subst: Optional[Dict[TypeVar, Type]] = None) -> Dict[TypeVar, Type]:
    """Unify two types and return a substitution mapping TypeVar->Type.

    Raises UnifyError on failure.
    """
    if subst is None:
        subst = {}

    t1 = apply_subst(subst, t1)
    t2 = apply_subst(subst, t2)

    if isinstance(t1, TypeVar):
        if t1 == t2:
            return subst
        if occurs_check(t1, t2, subst):
            raise UnifyError(f"occurs check failed: {t1} in {t2}")
        new = dict(subst)
        new[t1] = t2
        return new

    if isinstance(t2, TypeVar):
        return unify(t2, t1, subst)

    if isinstance(t1, TypeConst) and isinstance(t2, TypeConst):
        if t1.name == t2.name:
            return subst
        raise UnifyError(f"type mismatch: {t1} vs {t2}")

    if isinstance(t1, TypeArrow) and isinstance(t2, TypeArrow):
        s1 = unify(t1.left, t2.left, subst)
        s2 = unify(apply_subst(s1, t1.right), apply_subst(s1, t2.right), s1)
        return s2

    if isinstance(t1, TypeList) and isinstance(t2, TypeList):
        return unify(t1.elem, t2.elem, subst)

    raise UnifyError(f"cannot unify types {t1} and {t2}")


def generalize(env: Dict[str, Scheme], t: Type) -> Scheme:
    env_vars = set()
    for s in env.values():
        env_vars |= type_vars_of_scheme(s)
    free = type_vars_of_type(t) - env_vars
    return Scheme(list(free), t)


def instantiate(s: Scheme) -> Type:
    mapping: Dict[TypeVar, Type] = {v: fresh_type_var() for v in s.vars}
    return apply_subst(mapping, s.type)


def _demo_unify() -> None:
    # Quick sanity checks
    a = TypeVar('a')
    b = TypeVar('b')
    TInt = TypeConst('Int')
    TBool = TypeConst('Bool')

    s = unify(a, TInt)
    print('unify a ~ Int ->', s)

    s = unify(TypeList(a), TypeList(TInt))
    print('unify [a] ~ [Int] ->', s)

    # arrow unify
    s = unify(TypeArrow(a, TInt), TypeArrow(TInt, b))
    print('unify (a -> Int) ~ (Int -> b) ->', s)


if __name__ == '__main__':
    _demo_unify()
