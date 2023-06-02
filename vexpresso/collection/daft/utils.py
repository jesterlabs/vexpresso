import inspect

import daft


def convert_args(*args):
    return [
        arg.to_pylist() if isinstance(arg, daft.series.Series) else arg for arg in args
    ]


def convert_kwargs(**kwargs):
    return {
        k: kwargs[k].to_pylist()
        if isinstance(kwargs[k], daft.series.Series)
        else kwargs[k]
        for k in kwargs
    }


def transformation(object_or_function):
    def wraps(*args, **kwargs):
        args = convert_args(*args)
        kwargs = convert_kwargs(**kwargs)
        return object_or_function(*args, **kwargs)

    wraps.__signature__ = inspect.signature(object_or_function)
    _udf = daft.udf(return_dtype=daft.DataType.python())(wraps)
    return _udf


Transformation = transformation
