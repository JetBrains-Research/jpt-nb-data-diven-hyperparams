import ast
from common_util.const import NOT_A_MODEL_NAME


def get_func_name(node: ast.Call) -> str:
    # Assume that 'foo' can't be model name in case of such Call: foo(...)(...)
    # Only in case of such Call: foo(...)
    if not isinstance(node, ast.Call):
        raise TypeError(f'Expected ast.Call node, but {type(node).__name__} received')

    try:
        return node.func.id
    except AttributeError:
        try:
            return node.func.attr
        except AttributeError:
            return NOT_A_MODEL_NAME
