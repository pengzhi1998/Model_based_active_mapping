from __future__ import print_function
from __future__ import absolute_import
import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker
from pylint.checkers.utils import check_messages, decorated_with_property

MSGS = {
    'E2111': ('Properties are not allowed in shining software code',
              'no-properties',
              'Properties are not allowed in shining software code'),
}


class CheckIllegalConstructs(BaseChecker):
    """
    Check for some constructs prohibited in shining:
      - properties
    """

    __implements__ = IAstroidChecker

    name = 'shining_illegal_constructs'
    msgs = MSGS
    priority = -2
    options = ()

    @check_messages(*list(MSGS.keys()))
    def visit_functiondef(self, node):
        """triggered when an import statement is seen"""
        if isinstance(node, astroid.FunctionDef) and decorated_with_property(node):
            self.add_message('no-properties', node=node)


def register(linter):
    """required method to auto register this checker """
    linter.register_checker(CheckIllegalConstructs(linter))
