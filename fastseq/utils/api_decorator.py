import inspect
import logging
import sys

from functools import wraps

logger = logging.getLogger(__name__)


def get_class(method):
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
    cls = get_class(method)
    if not cls:
        raise ValueError(
            "Please input a valid class method instead of {}".format(
                method.__name__))

    def decorator(new_method):
        setattr(cls, method.__name__, new_method)

    return decorator


def add_method(cls):
    def decorator(func):
        logger.warning("A new method `{}`is added to `class {}`".format(
            func.__name__, cls.__name__))

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


def export_api(module_name, obj_name):
    def decorator(obj):
        delattr(sys.modules[module_name], obj_name)
        setattr(sys.modules[module_name], obj_name, obj)
        logger.info("Export api `{}.{}` for {}".format(module_name, obj_name,
                                                       obj))
        return obj

    return decorator


def replace(target_obj):
    def decorator(new_obj):
        for k, v in sys.modules.items():
            if (target_obj.__name__ in v.__dict__
                    and v.__dict__[target_obj.__name__] == target_obj):
                delattr(sys.modules[k], target_obj.__name__)
                setattr(sys.modules[k], target_obj.__name__, new_obj)
                logger.info("In module {}, {} is replaced by {}".format(
                    k, target_obj, new_obj))
        return new_obj

    return decorator
