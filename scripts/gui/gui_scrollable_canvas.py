import tkinter as tk
from scripts.gui.gui_utils import *

class TkScrollableCanvas(tk.Canvas):

    def __init__(self,
                 master):

        super().__init__()

        self.canvas = tk.Canvas(master)
        self.canvas_frame = tk.Frame(self.canvas)
        v_scroll = tk.Scrollbar(master, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=0, sticky="nse")

        self.canvas.create_window((0, 0), window=self.canvas_frame, anchor='nw')

        self.configure_scroll_region()

        self.canvas_frame.bind('<Enter>', self._bound_to_mousewheel)
        self.canvas_frame.bind('<Leave>', self._unbound_to_mousewheel)

    def configure_scroll_region(self):
        self.master.update()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bound_to_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbound_to_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
