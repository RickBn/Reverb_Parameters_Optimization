from scripts.gui.audio_widget import TkPyplot
from scripts.gui.audio_handler import TkAudioHandler
from scripts.gui.gui_utils import *


class TkGuiPlotComparison(tk.Frame):
    def __init__(self, master: tk.Frame, screen_width: int, screen_height: int):
        super().__init__()

        num_plots = 2

        init_rir_path = 'audio/input/chosen_rirs/'

        self.tk_plot, self.canvas, self.toolbar = (list() for _ in range(3))

        for i in range(num_plots):
            self.tk_plot.append(TkPyplot(master,
                                         fig_w=int(screen_width * 0.5),
                                         fig_h=int(screen_height * 0.75),
                                         row=3,
                                         col=i,
                                         sticky="n"))

            self.canvas.append(self.tk_plot[i].canvas)
            self.toolbar.append(self.tk_plot[i].toolbar)

            TkAudioHandler(master, default_audio_path=init_rir_path, row=2, col=i, sticky="n",
                           slave_widget=self.tk_plot[i])

            self.toolbar[i].grid(row=0, column=i, sticky='nsew')
            self.canvas[i].get_tk_widget().grid(row=1, column=i, sticky="nsew")
