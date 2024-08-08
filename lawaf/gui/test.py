from nicegui import ui


def gui_for_setting_params():
    params = dict(
        method="scdmk",  # options: scdmk/projected/dummy
        nwann=3,
        # selected_basis=[9, 10, 11],
        anchors={(0.0, 0.0, 0.0): (0, 1, 2)},
        use_proj=True,
        weight_func_params=(0, 100.010),
        weight_func="unity",
        kmesh=(4, 4, 4),
        gamma=True,
        kshift=(0.000, 0.000, 0.000),
        # enhance_Amn=-2,
    )
    ui.label("Lattice Wannier function")
    ui.select(
        ["scdmk", "projected", "dummy"],
        label="method",
        on_change=lambda x: params.update(method=x),
    )
    ui.number(
        "nwann", value=params["nwann"], on_change=lambda x: params.update(nwann=x)
    )
    ui.number("Anchors", value="0", on_change=lambda x: params.update(anchors=x))
    ui.checkbox(
        "Use projection",
        value=params["use_proj"],
        on_change=lambda x: params.update(use_proj=x),
    )
    ui.number(
        "Weight function params",
        value=10,
        on_change=lambda x: params.update(weight_func_params=x),
    )
    ui.select(
        ["unity", "gaussian", "exponential"],
        label="Weight function",
        value=params["weight_func"],
        on_change=lambda x: params.update(weight_func=x),
    )
    # ui.number("Kmesh", value=params["kmesh"], on_change=lambda x: params.update(kmesh=x))
    ui.checkbox(
        "Gamma", value=params["gamma"], on_change=lambda x: params.update(gamma=x)
    )

    ui.button("Run", on_click=lambda: print(params))

    ui.run()


if __name__ in {"__main__", "__mp_main__"}:
    gui_for_setting_params()
