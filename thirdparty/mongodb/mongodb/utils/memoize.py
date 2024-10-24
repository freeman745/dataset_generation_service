import functools


def make_hashable(obj):
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(e) for e in obj)
    elif isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in obj.items())
    else:
        return obj


class memoize:
    def __init__(self, func):
        self.func = func
        func_name = func.fget.__name__ if isinstance(func, property) else func.__name__
        self.name = "_" + func_name + "_memoized"
        self.cache = "_{}_cache".format(func_name)

    def __get__(self, instance, cls):
        if instance is None:
            return self
        if isinstance(self.func, property):
            return self.get_property_value(instance)
        return functools.partial(self.method_with_args, instance)

    def get_property_value(self, instance):
        if not hasattr(instance, self.name):
            value = self.func.fget(instance)
            setattr(instance, self.name, value)
        return getattr(instance, self.name)

    def method_with_args(self, instance, *args, **kwargs):
        cache = getattr(instance, self.cache, {})
        hashable_args = make_hashable(args)
        hashable_kwargs = make_hashable(kwargs)
        key = (hashable_args, hashable_kwargs)
        if key not in cache:
            cache[key] = self.func(instance, *args, **kwargs)
            setattr(instance, self.cache, cache)
        return cache[key]
