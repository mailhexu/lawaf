import sys

from lawaf.utils.port import get_unused_port
import numpy as np
from nicegui import ui

from lawaf.params import WannierParams

"""
    # default parameters.
    method = "scdmk"
    kmesh: Tuple[int] = (5, 5, 5)
    kshift = np.array([0, 0, 0], dtype=float)
    kpts = None
    kweights = None
    gamma: bool = True
    nwann: int = 0
    weight_func: Union[None, str, callable] = "unity"
    weight_func_params: Union[None, dict] = None
    selected_basis: Union[None, List[int]] = None
    anchors: Union[None, List[int]] = None
    anchor_kpt: Tuple[int] = (0, 0, 0)
    anchor_ibands: Union[None, List[int]] = None
    use_proj: bool = True
    proj_order: int = 1
    exclude_bands: Tuple[int] = ()
    sort_cols: bool = True
    enhance_Amn: int = 0
    selected_orbdict = None
    orthogonal = True
"""


class ParamsGui:
    def __init__(self, port=None):
        # default parameters
        self.params = WannierParams()
        self.port = get_unused_port(default=port)

    def run(self):
        self.create_ui()

    def create_ui(self):
        ui.add_css(
            """
        :root {
        --nicegui-default-padding: 0.1rem;
        --nicegui-default-gap: 0.5rem;
        }
        """
        )
        ui.label("LaWaF")
        self.params.method = "scdmk"
        # number of wannier functions
        with ui.row():
            self.parameters_card()
            self.figure_card()
        ## projection order
        # ui.number("Projection order", value=self.params.proj_order, on_change=lambda x: self.params.set("proj_order", int(x.value)))
        ui.run(port=self.port)

    def plot_band(self, ax, pl):
        x = np.linspace(0.1, 10, 100)
        y = np.cos(np.random.random(len(x)))
        ax.clear()
        ax.scatter(x, y)
        pl.update()

    def wannierize(self, ax, pl):
        ax.clear()
        ax.scatter([1, 1], [2, 3])
        pl.update()

    def figure_card(self):
        with ui.card():
            with ui.row():
                ui.button("Plot band", on_click=lambda: self.plot_band(ax, pl=pl))
                ui.button("Wannierize", on_click=lambda: self.wannierize(ax, pl=pl))
                ui.button(
                    "Save figure",
                    on_click=lambda: ui.notify(f"self.params={self.params}"),
                )
            with ui.row():
                pl = ui.matplotlib()  # .classes("w-1/2, h-100, border")
                fig = pl.figure
                ax = fig.gca()
                # fig.tight_layout()

                # min_max_range = ui.range(min=0, max=100, value={'min': 20, 'max': 50}).classes(" wf, rotate-90")

                # with ui.card():#.classes("p-3,w-1, h-100"):
                # min_max_range = ui.range(min=0, max=100, value={'min': 20, 'max': 50}).classes("vertical-center, border, rotate-90")
                min_max_range = ui.range(
                    min=0, max=100, value={"min": 20, "max": 50}
                ).classes("vertical-top, border")

                # with ui.container():
                ui.label().bind_text_from(
                    min_max_range,
                    "value",
                    backward=lambda v: f'min: {v["min"]}, max: {v["max"]}',
                )

    def parameters_card(self):
        with ui.card():
            ui.number(
                "number of wannier functions",
                value=self.params.nwann,
                on_change=lambda x: self.params.set("nwann", int(x.value)),
            )  # .classes("w-full")
            with ui.row():
                ui.label("K-mesh")
                ui.input(
                    " 2,2,2",
                    on_change=lambda x: self.params.set(
                        "kmesh", [int(i) for i in x.value.split(",")]
                    ),
                )
                # ui.checkbox("Gamma", on_change=lambda x: self.params.set("gamma", x.value), value = self.params.gamma)

            ui.separator()
            with ui.tabs().classes("w-full") as tabs:
                scdmk_tab = ui.tab("SCDM-k")
                PWF_tab = ui.tab("PWF")
            with ui.tab_panels(
                tabs,
                value=scdmk_tab,
                on_change=lambda x: self.params.set("method", x.value),
            ):  # as method_panels:
                with ui.tab_panel(scdmk_tab):
                    self.scdmk_tab()
                with ui.tab_panel(PWF_tab):
                    self.PWF_tab()

        # ui.select(["scdmk", "projected"], label="Method", value=self.params.method,
        #          on_change=lambda x: self.params.set("method", x.value))
        # with ui.card():

    def PWF_tab(self):
        with ui.row():
            ui.label("projector type")
            proj_type = ui.toggle(
                ["atomic", "mode"],
                value="mode",
                on_change=lambda x: self.params.set("proj_type", x.value),
            )

        with ui.column().bind_visibility_from(
            proj_type, "value", backward=lambda x: x == "mode"
        ):
            with ui.row():
                ui.label("mode kpoint")
                ui.input(
                    "0.0,0.0,0.0",
                    on_change=lambda x: self.params.set(
                        "anchor_kpt", [float(i) for i in x.value.split(",")]
                    ),
                )
            # anchor bands
            with ui.row():
                ui.label("mode band index")
                ui.input(
                    "0,1,2",
                    on_change=lambda x: (
                        self.params.set(
                            "anchor_ibands", [int(i) for i in x.value.split(",")]
                        ),
                        ui.notify(f"self.params={self.params}"),
                    ),
                )

        with ui.column().bind_visibility_from(
            proj_type, "value", backward=lambda x: x == "atomic"
        ):
            with ui.row():
                ui.label("atomic index")
                ui.input(
                    "0,1,2",
                    on_change=lambda x: self.params.set(
                        "selected_basis", [int(i) for i in x.value.split(",")]
                    ),
                )

    def scdmk_tab(self):
        # ui.select(["unity", "Gaussian", "Fermi", "range"], label="Weight function", value=self.params.weight_func,
        #          on_change=lambda x: self.params.set("weight_func", x.value))

        # anchor kpoints
        # use projection
        ui.checkbox(
            "Use projection",
            on_change=lambda x: self.params.set("use_proj", x.value),
            value=self.params.use_proj,
        )

        scdmk_mode = ui.toggle(["manaul", "auto", "columns"], value="manaul")
        with ui.column().bind_visibility_from(
            scdmk_mode, "value", backward=lambda x: x == "manaul"
        ):
            with ui.row().bind_visibility_from(
                scdmk_mode, "value", backward=lambda x: x == "manaul"
            ):
                ui.label("Anchor kpoint")
                ui.input(
                    "0.0,0.0,0.0",
                    on_change=lambda x: self.params.set(
                        "anchor_kpt", [float(i) for i in x.value.split(",")]
                    ),
                )
            # anchor bands
            with ui.row():
                ui.label("Anchor band indices")
                ui.input(
                    "0,1,2",
                    on_change=lambda x: (
                        self.params.set(
                            "anchor_ibands", [int(i) for i in x.value.split(",")]
                        ),
                        ui.notify(f"self.params={self.params}"),
                    ),
                )

        with ui.column().bind_visibility_from(
            scdmk_mode, "value", backward=lambda x: x == "columns"
        ):
            # selected basis
            with ui.row():
                ui.label("Selected columns")
                ui.input(
                    "0,1,2",
                    on_change=lambda x: self.params.set(
                        "selected_basis", [int(i) for i in x.value.split(",")]
                    ),
                )


if __name__ in ["__main__", "__mp_main__"]:
    pg = ParamsGui(port=int(sys.argv[1]) if len(sys.argv) > 1 else None)
    pg.run()
