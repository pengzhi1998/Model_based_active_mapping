from __future__ import print_function
from __future__ import absolute_import
import astroid
from pylint.interfaces import IAstroidChecker
from pylint.checkers import BaseChecker
from pylint.checkers.utils import check_messages


MSGS = {
    'E1113': ('Import inside a function from shining is not allowed',
              'infunction-import',
              'Import inside a function from shining is not allowed'),
    'E1114': ('Imports from catkin are not allowed',
              'catkin-import',
              'Imports from catkin are not allowed'),
    'E1115': ('Imports from src/calibration are not allowed',
              'calibration-import',
              'Imports from src/calibration are not allowed'),
    'E1116': ('Imports from tests are not allowed',
              'test-import',
              'Imports from tests are not allowed'),
    'E1117': ('Imports from sandboxes are not allowed',
              'sandbox-import',
              'Imports from sandboxes are not allowed'),
}


class CheckIllegalImports(BaseChecker):
    """
    Check for
      - imports inside functions
      - imports from catkin
      - imports from calibration scripts folder
    """

    __implements__ = IAstroidChecker

    name = 'shining_illegal_imports'
    msgs = MSGS
    priority = -2
    options = ()

    @check_messages(*list(MSGS.keys()))
    def visit_import(self, node):
        """triggered when an import statement is seen"""
        self._check_node(node.names[0][0], node)

    @check_messages(*list(MSGS.keys()))
    def visit_importfrom(self, node):
        """triggered when a from statement is seen"""
        self._check_node(node.modname, node)

    def _check_node(self, name, node):
        if name.startswith('catkin_ws'):
            self.add_message('catkin-import', node=node)
            return

        if name.startswith('calibration'):
            self.add_message('calibration-import', node=node)
            return

        if not name.startswith('shining_software'):
            return

        module_name = name.split('.')[-1]
        if module_name.startswith('test_'):
            self.add_message('test-import', node=node)

        if 'sandbox' in module_name:
            self.add_message('sandbox-import', node=node)

        if _is_in_function_import(node):
            self.add_message('infunction-import', node=node)


def _is_in_function_import(node):
    '''
    Determines that import is a module-level import
    :param node: astroid node
    :return: True if an import is inside a function
    '''
    parent = node.parent
    allowed_node_types = (astroid.TryExcept, astroid.TryFinally, astroid.If)
    while not isinstance(parent, astroid.Module):
        if not isinstance(parent, allowed_node_types):
            return True
        parent = parent.parent
    return False


def register(linter):
    """required method to auto register this checker """
    linter.register_checker(CheckIllegalImports(linter))
