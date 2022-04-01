from __future__ import print_function, absolute_import, division

import bc_exploration.cpp


def test_bindings(debug=False):
    assert bc_exploration.cpp.__doc__
    assert bc_exploration.cpp.c_astar.__doc__
    assert bc_exploration.cpp.c_oriented_astar.__doc__
    assert bc_exploration.cpp.c_get_astar_angles.__doc__
    assert bc_exploration.cpp.c_check_for_collision.__doc__

    if debug:
        print(bc_exploration.cpp.__doc__)
        print(bc_exploration.cpp.c_astar.__doc__)
        print(bc_exploration.cpp.c_oriented_astar.__doc__)
        print(bc_exploration.cpp.c_get_astar_angles.__doc__)
        print(bc_exploration.cpp.c_check_for_collision.__doc__)


if __name__ == '__main__':
    test_bindings(debug=True)
