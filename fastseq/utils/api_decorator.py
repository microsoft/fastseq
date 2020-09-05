# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Decorators for revising the code in runtime.
"""

import inspect
import sys

from types import ModuleType
from functools import wraps

from fastseq.logging import get_logger

logger = get_logger(__name__)

FAIRSEQ_OPTIMIZED_CLASSES = []

def register_fairseq_optimized_class(optimized_class):
    """ A decorator used to register all the optimized classes

    Args:
        optimized_class (class): the optimized class.

    Returns:
        return the input optimized class.
    """
    FAIRSEQ_OPTIMIZED_CLASSES.append(optimized_class)
    return optimized_class


def get_class(method):
    """Get the class of the input unbound method.

    Args:
        method (object): an unbound method or function.

    Returns:
        A class of the input method. It will be `None` if the input method is
        a function.
    """
    if inspect.ismethod(method):
        for cls in inspect.getmro(method.__self__.__class__):
            if cls.__dict__.get(method.__name__) is method:
                return cls
        method = method.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(method):
        cls = getattr(
            inspect.getmodule(method),
            method.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return getattr(method, '__objclass__',
                   None)  # handle special descriptor objects


def override_method(method):
    """A decorator to override the unbound method.

    Due to the limitation of `get_class()`, the class which contains the input
    unbound method can not be defined inside another class.

    Example:

    ```python
    class MyClass:
        def f(self):
            return 'hello world'
    @override_method(MyClass.f)
    def f(self):
      print 'hello world'
    ```

    Args:
        method (object): the unbound method to be overried.

    Raises:
        ValueError: if `method` is not a class unbound method.

    Returns:
        A decorator function to override the unbound method.
    """
    cls = get_class(method)
    if not cls:
        raise ValueError(
            "Please input a valid class unbound method instead of {}".format(
                method.__name__))

    def decorator(new_method):
        logger.debug("The method `{}`is replaced by `{}`".format(
            method.__qualname__, new_method.__qualname__))

        @wraps(new_method)
        def wrapper(self, *args, **kwargs):
            return new_method(self, *args, **kwargs)

        setattr(cls, method.__name__, wrapper)
        return method

    return decorator


def add_method(cls):
    """A decorator to add a new unbound method to the specified class.

    Due to the limitation of `get_class()`, the class which contains the input
    unbound method can not be defined inside another class.

    Example:

    ```python
    class MyClass:
        def f(self):
            return 'hello world'

    @add_method(MyClass)
    def print(self):
      print 'hello world'
    ```

    Args:
        cls (class): the class to be added with a new method.

    Returns:
        A decorator function to add the unbound method.

    """
    def decorator(method):
        logger.debug("A new method `{}`is added to `class {}`".format(
            method.__name__, cls.__name__))

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            return method(self, *args, **kwargs)

        setattr(cls, method.__name__, wrapper)
        return method

    return decorator


def export_api(module_name, obj_name):
    """A decorator to export the API.

    Example:

    ```python
    @export_api('test.export.api', 'MyClass')
    class MyClass:
        def f(self):
            return 'hello world'
    ```

    Args:
        module_name (str): the module name
        obj_name (str): the object name, which can be the class name or
                           function name.

    Returns:
        A decorator function to export the API.
    """
    if not module_name in sys.modules:
        fake_module = ModuleType(module_name)
        sys.modules[module_name] = fake_module

    def decorator(obj):
        if hasattr(sys.modules[module_name], obj_name):
            delattr(sys.modules[module_name], obj_name)
        setattr(sys.modules[module_name], obj_name, obj)
        logger.debug("Export {} as `{}.{}`.".format(obj, module_name, obj_name))
        return obj

    return decorator


def replace(target_obj):
    """A decorator to replace the specified obj.

    `target_obj` can be a class or a function.

    Example:

    ```python
    class A:
        def f(self):
            print('class A')
    @replace(A)
    class B:
        def f(self):
            print('class B')
    ```

    Args:
        target_obj (class/func/method): a class, method, or function to be
                                        replaced.

    Returns:
        A decorator function to replace the input object.
    """
    def decorator(new_obj):
        for k, v in sys.modules.items():
            if (target_obj.__name__ in v.__dict__
                and v.__dict__[target_obj.__name__] is target_obj):
                delattr(sys.modules[k], target_obj.__name__)
                setattr(sys.modules[k], target_obj.__name__, new_obj)
                logger.debug("In module {}, {} is replaced by {}".format(
                    k, target_obj, new_obj))
            # replace target_obj if it is used as the base classes.
            for key in list(v.__dict__.keys()):
                if (inspect.isclass(v.__dict__[key]) and
                    v.__dict__[key] != new_obj and
                    target_obj in v.__dict__[key].__bases__):
                    idx = v.__dict__[key].__bases__.index(target_obj)
                    bases = list(v.__dict__[key].__bases__)
                    bases[idx] = new_obj
                    v.__dict__[key].__bases__ = tuple(bases)
                    logger.debug(
                        "In module {}, the base class of {} is replaced by {}"
                        .format(k, v.__dict__[key], new_obj))
        return new_obj

    return decorator
