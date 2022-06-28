import tkinter as tk


def configure_grid_all(widget: tk.Widget):
    for col_num in range(widget.grid_size()[0]):
        widget.columnconfigure(col_num, weight=1)

    for row_num in range(widget.grid_size()[1]):
        widget.rowconfigure(row_num, weight=1)
