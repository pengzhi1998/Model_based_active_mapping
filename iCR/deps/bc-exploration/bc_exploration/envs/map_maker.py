"""map_maker.py
Allows user to make a map for environment testing, gui window
"""

from __future__ import print_function, absolute_import, division

import os
import argparse
import cv2
import numpy as np

from functools import partial

try:
    # for Python2
    from Tkinter import Canvas, Tk, Label, Entry, LEFT, X, Button
except ImportError:
    # for Python3
    from tkinter import Canvas, Tk, Label, Entry, LEFT, X, Button

from bc_exploration.utilities.paths import get_maps_dir


class Cell:
    """
    Building block for the gui grid. It can visualize itself, and toggle filled, not filled
    """
    FILLED_COLOR_BG = "black"
    EMPTY_COLOR_BG = "white"
    FILLED_COLOR_BORDER = "black"
    EMPTY_COLOR_BORDER = "grey"

    def __init__(self, window, row, column, size, is_filled=False):
        """
        Building block for the gui grid. It can visualize itself, and toggle filled, not filled
        :param window Tk: the Tk application object
        :param row int: the row index for the cell
        :param column int: the column index for the cell
        :param size int: the size in px the cell should take on the screen
        :param is_filled bool: whether or not the cell is filled
        """
        self.window = window
        self.row = row
        self.column = column
        self.size = size
        self.is_filled = is_filled

    def toggle_filled(self):
        """
        Toggle is_filled bool variable
        """
        self.is_filled = not self.is_filled

    def draw(self):
        """
        Visualize the cell in the window, with appropriate colors defined above the class def.
        """
        if self.window is not None:
            background_color = Cell.EMPTY_COLOR_BG
            border_color = Cell.EMPTY_COLOR_BORDER
            if self.is_filled:
                background_color = Cell.FILLED_COLOR_BG
                border_color = Cell.FILLED_COLOR_BORDER

            cell_range_row = (self.row * self.size, self.row * self.size + self.size)
            cell_range_col = (self.column * self.size, self.column * self.size + self.size)

            self.window.create_rectangle(cell_range_col[0], cell_range_row[0],
                                         cell_range_col[1], cell_range_row[1],
                                         fill=background_color,
                                         outline=border_color)


class Grid(Canvas):  # pylint: disable=too-many-ancestors
    """
    Grid object that will allow drawing on it by changing the underlying cell values
    """
    def __init__(self, window, num_rows, num_columns, cell_size):
        """
        Defines a grid-like composition of cells, visualizable editable grid for drawing maps
        :param window Tk: the Tk application object
        :param num_rows int: number of rows desired
        :param num_columns int: number of columns desired
        :param cell_size int: the size in px the cell should take on the screen
        """
        Canvas.__init__(self,
                        master=window,
                        width=cell_size * num_columns,
                        height=cell_size * num_rows)

        self.num_columns = num_columns
        self.num_rows = num_rows
        self.cell_size = cell_size

        # initialize grid
        self.grid = []
        for r in range(num_rows):
            row = []
            for c in range(num_columns):
                if r == 0 or r == num_rows - 1 or c == 0 or c == num_columns - 1:
                    row.append(Cell(self, row=r, column=c, size=cell_size, is_filled=True))
                else:
                    row.append(Cell(self, row=r, column=c, size=cell_size, is_filled=False))
            self.grid.append(row)

        self.switched = []

        # bind click action
        self.bind("<Button-1>", self.handle_mouse_click)
        # bind moving while clicking
        self.bind("<B1-Motion>", self.handle_mouse_motion)
        # bind release button action - clear the memory of modified cells.
        self.bind("<ButtonRelease-1>", self.handle_button_release)

        self.draw()

    def draw(self):
        """
        Draw each cell in the grid
        """
        for row in self.grid:
            for cell in row:
                cell.draw()

    def _get_coords(self, event):
        """
        Get the row, column of the cell that triggered the event
        :param event Event: mouse click event
        :return Tuple(int): row, column
        """
        row = int(event.y / self.cell_size)
        column = int(event.x / self.cell_size)
        return row, column

    def handle_button_release(self, _):
        """
        Button release handler
        :param _ Event: event object, not used
        """
        self.switched = []

    def handle_mouse_click(self, event):
        """
        Mouse click handler for the editor
        :param event Event: mouse click event
        """
        row, column = self._get_coords(event=event)
        if row > self.num_rows or row < 0 or column > self.num_columns or column < 0:
            return

        cell = self.grid[row][column]
        cell.toggle_filled()
        cell.draw()

        self.switched.append(cell)

    def handle_mouse_motion(self, event):
        """
        Mouse drag handler
        :param event Event: mouse drag event
        """
        row, column = self._get_coords(event=event)
        if row > self.num_rows or row < 0 or column > self.num_columns or column < 0:
            return

        cell = self.grid[row][column]

        if cell not in self.switched:
            cell.toggle_filled()
            cell.draw()
            self.switched.append(cell)

    def save(self, filename):
        """
        Save map to file
        :param filename str: filename of where to save the map
        """
        occupany_map = []
        for row in self.grid:
            map_row = []
            for cell in row:
                map_row.append(255 * int(not cell.is_filled))
            occupany_map.append(map_row)

        cv2.imwrite(filename, np.array(occupany_map))


def save_button_callback(grid, entry, directory='.'):
    """
    Callback for the save button
    :param grid Grid: grid object
    :param entry Entry: text field object which has the name of the file
    :param directory str: where to save it
    """
    filename = entry.get()

    if ".png" not in filename:
        filename += ".png"

    if not os.path.exists(directory):
        os.makedirs(directory)

    grid.save(os.path.join(directory, filename))


def make_map(num_rows, num_columns, cell_size, save_dir='.'):
    """
    Main interface for making a map
    :param num_rows int: number of rows desired
    :param num_columns int: number of columns desired
    :param cell_size int: the size in px the cell should take on the screen
    :param save_dir str: directory of where to save the map
    """
    app = Tk(className="map maker")
    app.title("Map Maker")
    grid = Grid(app, num_rows=num_rows, num_columns=num_columns, cell_size=cell_size)
    grid.pack()

    Label(app, text="Filename: ").pack(side=LEFT)

    text_field = Entry(app, text="")
    text_field.pack(side=LEFT, fill=X, expand=True)

    save_button = Button(app, text="Save", command=partial(save_button_callback,
                                                           grid=grid,
                                                           entry=text_field,
                                                           directory=save_dir))
    save_button.pack(side=LEFT)

    app.mainloop()


# todo add load and edit and view
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-num_rows', type=int, help='number of rows', default=11)
    parser.add_argument('-num_columns', type=int, help='number of columns', default=11)
    parser.add_argument('-cell_size', type=int, help='cell size in pixels', default=30)
    parser.add_argument('-save_dir', type=str, help='cell size in pixels',
                        default=get_maps_dir())

    args = parser.parse_args()
    make_map(num_rows=args.num_rows,
             num_columns=args.num_columns,
             cell_size=args.cell_size, save_dir=args.save_dir)
