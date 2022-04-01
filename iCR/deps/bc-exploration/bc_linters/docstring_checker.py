"""Braincorp doc string checker. """
from __future__ import print_function
from __future__ import absolute_import

import re
import sys
import astroid
from collections import OrderedDict
from pylint.checkers.base import _BasicChecker
from pylint.checkers.utils import check_messages, get_node_last_lineno, safe_infer
from collections import namedtuple

NO_REQUIRED_DOC_RGX = re.compile('^_')
PY3K = sys.version_info >= (3, 0)


MSGS = {
    "C9111": (
        "Missing %s docstring",  # W0131
        "bc-missing-docstring",
        "Used when a module, function, class or method has no docstring."
        "Some special methods like __init__ doesn't necessary require a "
        "docstring.",
    ),
    "C9112": (
        "Empty %s docstring",  # W0132
        "bc-empty-docstring",
        "Used when a module, function, class or method has an empty "
        "docstring (it would be too easy ;).",
    ),
    "C9113": (
        "Argument %s after return %s in docstring",
        "argument-after-return-docstring",
        ":param definition after :return is not allowed.",
    ),
    "C9114": (
        "Another return %s after return %s in docstring",
        "return-after-return-docstring",
        ":return definition after another :return is not allowed.",
    ),
    "C9115": (
        "Empty or too short (<6 characters) docstring description of a function",
        "empty-function-description-docstring",
        "There has to be non-empty function description (>5 characters)",
    ),
    "C9116": (
        "Argument '%s' is not documented (or description is too short <6 characters)",
        "missing-argument-docstring",
        "Use :param <name> <type>: <description> format",
    ),
    "C9117": (
        "Documented argument '%s' does not exist",
        "extra-argument-docstring",
        "Remove unnecessary arguments from docstrings",
    ),
    "C9118": (
        "Function's return value is not documented (or description is too short)",
        "no-return-docstring",
        "Add :return <type>: <description> in the doc string",
    ),
    "C9119": (
        "No need for :return: docstring since the function doesn't have return statement ",
        "unnecessary-return-docstring",
        "Remove :return: from the doc string, since function doesn't have return statement",
    ),
    "C9120": (
        "Order of arguments in the docstring is wrong",
        "wrong-parameter-order-docstring",
        "Parameters should be documented in the same order as they received by the function",
    ),
    "C9121": (
        "Argument '%s' is documented multiple times",
        "duplicate-argument-docstring",
        "Every argument has to be documented only once",
    )
}


ReturnDocstring = namedtuple('ReturnDocstring', [
    'doc_line',  # String, full line of the docstring
    'type',  # String, Documented type
    'description',  # String, description
])


class DocstringChecker(_BasicChecker):
    """Checks a many semantic properties of a docstring of functions,
    such as correspondence of parameters and return values to the documented ones
    Initial code was inspired by
    https://github.caom/PyCQA/pylint/blob/fcc01516ae176ad3fdedc4497328105f3314e376/pylint/checkers/base.py
    """

    msgs = MSGS
    options = (
        (
            "bc-no-docstring-rgx",
            {
                "default": NO_REQUIRED_DOC_RGX,
                "type": "regexp",
                "metavar": "<regexp>",
                "help": "Regular expression which should only match "
                "function or class names that do not require a "
                "docstring.",
            },
        ),
        (
            "bc-has-to-have-docstring-rgx",
            {
                "default": "",
                "type": "regexp",
                "metavar": "<regexp>",
                "help": "Regular expression which should match "
                        "function or class names that has to have "
                        "docstring, even if bc-no-docstring-rgx say otherwise",
            },
        ),
        (
            "bc-docstring-min-length",
            {
                "default": -1,
                "type": "int",
                "metavar": "<int>",
                "help": (
                    "Minimum line length for functions/classes that"
                    " require docstrings, shorter ones are exempt."
                ),
            },
        ),
    )

    @check_messages(*list(MSGS.keys()))
    def visit_module(self, node):  # pylint: disable=bc-missing-docstring
        self._check_docstring("module", node)

    @check_messages(*list(MSGS.keys()))
    def visit_classdef(self, node):  # pylint: disable=bc-missing-docstring
        if self.config.bc_no_docstring_rgx.match(node.name) is None:
            self._check_docstring("class", node)

    @staticmethod
    def _is_setter_or_deleter(node):  # pylint: disable=bc-missing-docstring
        names = {"setter", "deleter"}
        for decorator in node.decorators.nodes:
            if isinstance(decorator, astroid.Attribute) and decorator.attrname in names:
                return True
        return False

    @check_messages(*list(MSGS.keys()))
    def visit_functiondef(self, node):  # pylint: disable=bc-missing-docstring
        no_docstring = self.config.bc_no_docstring_rgx.match(node.name) is not None
        has_to_have_docstring = self.config.bc_has_to_have_docstring_rgx.match(node.name) is not None
        if has_to_have_docstring or not no_docstring:
            ftype = "method" if node.is_method() else "function"
            if node.decorators and self._is_setter_or_deleter(node):
                return

            if isinstance(node.parent.frame(), astroid.ClassDef):
                overridden = False
                # check if node is from a method overridden by its ancestor
                for ancestor in node.parent.frame().ancestors():
                    if node.name in ancestor and isinstance(
                        ancestor[node.name], astroid.FunctionDef
                    ):
                        overridden = True
                        break
                is_constructor = node.name == '__init__'
                self._check_docstring(
                    ftype, node,
                    report_missing=not overridden or is_constructor,
                    is_constructor=is_constructor
                )
            elif isinstance(node.parent.frame(), astroid.Module):
                self._check_docstring(ftype, node)
            else:
                return

    visit_asyncfunctiondef = visit_functiondef

    def _check_docstring(self, node_type, node, report_missing=True, is_constructor=False):
        """Check function and module docstring to comply with standards
        :param node_type String: string type of the node (e.g. "module")
        :param node astroid.Node: ast node of the cheking
        :param report_missing bool: whether to error when docstring is missing
        :param is_constructor bool: whether the node is a constructor or not
        """
        docstring = node.doc
        if docstring is None:
            if not report_missing:
                return
            if is_constructor and _extract_node_arg_names(node) == ['self']:
                # its ok to have empty contstructor doc if there are no arguments
                return
            lines = get_node_last_lineno(node) - node.lineno

            if node_type == "module" and not lines:
                # If the module has no body, there's no reason
                # to require a docstring.
                return
            max_lines = self.config.bc_docstring_min_length

            if node_type != "module" and max_lines > -1 and lines < max_lines:
                return

            if node.body and isinstance(node.body[0], astroid.Expr) and isinstance(node.body[0].value, astroid.Call):
                # Most likely a string with a format call. Let's see.
                func = safe_infer(node.body[0].value.func)
                if isinstance(func, astroid.BoundMethod) and isinstance(
                    func.bound, astroid.Instance
                ):
                    # Strings in Python 3, others in Python 2.
                    if PY3K and func.bound.name == "str":
                        return
                    if func.bound.name in ("str", "unicode", "bytes"):
                        return

            self.add_message(
                "bc-missing-docstring", node=node, args=(node_type,)
            )
        elif not docstring.strip():
            self.add_message(
                "bc-empty-docstring", node=node, args=(node_type,)
            )
        else:
            if node_type == 'method':
                # Check if the "@staticmethod" decorator exists
                is_static_method = any(['staticmethod' in decoratorname for decoratorname in node.decoratornames()])
                is_abstract_method = any(['abstractmethod' in decoratorname for decoratorname in node.decoratornames()])
                if is_static_method:
                    self._check_braincorp_docstring(docstring, _extract_node_arg_names(node), node,
                                                    check_function_description=True,
                                                    enforce_return_consistency=not is_abstract_method)
                else:
                    self._check_braincorp_docstring(docstring, _extract_node_arg_names(node)[1:], node,
                                                    check_function_description=not is_constructor,
                                                    enforce_return_consistency=not is_abstract_method)
            elif node_type == 'function':
                self._check_braincorp_docstring(docstring, _extract_node_arg_names(node), node,
                                                check_function_description=True,
                                                enforce_return_consistency=True)

    def _check_braincorp_docstring(self, docstring, arguments, node, check_function_description,
                                   enforce_return_consistency):
        '''
        Check that docstrings complies to the format (see below)
        :param docstring string: docstring
        :param arguments Collection[string]: list of argument names according to function definition
        :param node astroid.Node: node from ast
        :param check_function_description bool: whether to check function description length
        :param enforce_return_consistency bool: whether to check that return documentation is consistent with the code
        '''
        minimal_description_length = 6
        description = []
        total_description_length = 0
        return_statement = None
        argument_pattern = re.compile(r":param ([a-zA-Z0-9_]*)( *)(.*):( *)(.*)")
        parsed_arguments = OrderedDict()
        return_pattern = re.compile(r":return( *)(.*):( *)(.*)")
        for l in docstring.splitlines():
            if len(l.strip()) == 0:
                continue
            argument_result = argument_pattern.match(l.strip())
            if argument_result is not None:
                if return_statement is not None:
                    self.add_message(
                        "argument-after-return-docstring", node=node, args=(argument_result.group(1), return_statement.doc_line)
                    )
                if check_function_description and total_description_length < minimal_description_length:
                    self.add_message(
                        "empty-function-description-docstring", node=node
                    )
                    return
                argument_name = argument_result.group(1)
                if argument_name in parsed_arguments:
                    self.add_message(
                        "duplicate-argument-docstring", node=node, args=(argument_name,)
                    )
                else:
                    parsed_arguments[argument_name] = (l.strip(), argument_result.group(3), argument_result.group(5))
            else:
                return_result = return_pattern.match(l.strip())
                if return_result is not None:
                    if return_statement is not None:
                        self.add_message(
                            "return-after-return-docstring", node=node, args=(l.strip(), return_statement.doc_line)
                        )
                    if check_function_description and total_description_length < minimal_description_length:
                        self.add_message(
                            "empty-function-description-docstring", node=node
                        )
                        return
                    return_statement = ReturnDocstring(l.strip(), return_result.group(2), return_result.group(4))
                else:
                    description.append(l.strip())
                    total_description_length += len(l.strip())

        for a in arguments:
            if a not in parsed_arguments or (a in parsed_arguments and len(parsed_arguments[a][2]) < minimal_description_length):
                self.add_message(
                    "missing-argument-docstring", node=node, args=(a,)
                )
        for a in parsed_arguments:
            if a not in arguments:
                self.add_message(
                    "extra-argument-docstring", node=node, args=(a,)
                )

        if set(arguments) == set(parsed_arguments.keys()) and list(arguments) != list(parsed_arguments.keys()):
            self.add_message(
                "wrong-parameter-order-docstring", node=node
            )

        if enforce_return_consistency:
            has_return = node_has_non_empty_return(node)
            if has_return and (return_statement is None or len(return_statement.description) < minimal_description_length):
                self.add_message(
                    "no-return-docstring", node=node
                )

            if not has_return and return_statement is not None:
                self.add_message(
                    "unnecessary-return-docstring", node=node
                )


def node_has_non_empty_return(node):
    '''
    Whether a node has a return statement without any result
    :param node astroid.Node: ast node
    :return Bool: True if node has "return smth" in it
    '''
    if isinstance(node, astroid.Return) and node.value is not None:
        return True
    for child in node.get_children():
        if node_has_non_empty_return(child):
            return True
    return False


def _is_inner_function(node):
    """Check whether node is an inner function
    Walk up the node tree to check if a parent function exists
    :param node astroid.Node: ast node
    :return Bool: True if node is inner function
    """
    if isinstance(node, astroid.FunctionDef):
        return True

    if node.parent:
        return _is_inner_function(node.parent)

    return False


def _extract_node_arg_names(node):
    """
    Extract arguments from a node that represents a function
    :param node astroid.node: function of method ast node
    :return Collection[String]: argument names
    """
    args = node.args
    normal_args = [a.name for a in args.args]
    if args.kwarg is not None:
        normal_args.append(args.kwarg)
    if args.vararg is not None:
        normal_args.append(args.vararg)
    return normal_args


def register(linter):
    """required method to auto register this checker
    :param linter object: linter instance
    """
    linter.register_checker(DocstringChecker(linter))
