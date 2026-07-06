import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------------------------------------------------
# THEME / COLORS
# ----------------------------------------------------------------------
BG          = "#0d1117"   # window / figure background
PANEL_BG    = "#12161c"
FIELD_BG    = "#1c2129"
BORDER      = "#30363d"
AXIS_LINE   = "#8b949e"
GRID_MAJOR  = "#30363d"
GRID_MINOR  = "#20262d"
TEXT_COLOR  = "#e6edf3"
MUTED_TEXT  = "#8b949e"
ACCENT      = "#58a6ff"
TOGGLE_OFF  = "#30363d"
ERROR_COLOR = "#ff7b72"

OBJECT_COLOR        = "#00e5ff"   # cyan
IMAGE_REAL_COLOR    = "#ff5c5c"   # red
IMAGE_VIRTUAL_COLOR = "#ffd93d"   # yellow
ELEMENT_LENS_COLOR   = "#58a6ff"  # blue
ELEMENT_MIRROR_COLOR = "#d2d8de"  # silver
FOCUS_COLOR   = "#ffa657"         # orange
CENTER_COLOR  = "#ff5c5c"         # red
RAY_COLORS    = ["#7ee787", "#ffa657", "#d2a8ff"]  # green / orange / purple
INFINITY_RAY_COLOR = "#7ee787"
DIM_U_COLOR   = "#7ee787"
DIM_V_COLOR   = "#ff9dcc"
DIM_F_COLOR   = "#ffd93d"

ELEMENT_TYPES = ["Convex Lens", "Concave Lens", "Concave Mirror", "Convex Mirror"]

PRESETS = [
    "Custom (drag me)",
    "At Infinity",
    "Beyond 2F / C",
    "At 2F / C",
    "Between F and 2F/C",
    "At F",
    "Between P/O and F",
]

APP_AUTHOR = "Ayan Harsh"
APP_EMAIL = "contactayanharsh601@gmail.com"


def preset_to_x(preset, f_mag):
    """Return the object x-position (negative, left of element) for a preset."""
    table = {
        "Beyond 2F / C": -3.0 * f_mag,
        "At 2F / C": -2.0 * f_mag,
        "Between F and 2F/C": -1.5 * f_mag,
        "At F": -1.0 * f_mag,
        "Between P/O and F": -0.5 * f_mag,
    }
    return table.get(preset, None)


# ----------------------------------------------------------------------
# PHYSICS
# ----------------------------------------------------------------------
def signed_focal_length(element_type, f_mag):
    if element_type in ("Convex Lens", "Convex Mirror"):
        return f_mag
    else:
        return -f_mag


def compute_image(element_type, u, f):
    """
    Returns dict with v, m, real(bool), erect(bool), infinite(bool)
    u : object distance (signed, negative for real object on the left)
    f : signed focal length
    """
    is_lens = "Lens" in element_type
    result = {"infinite": False, "v": None, "m": None, "real": None, "erect": None}

    if is_lens:
        denom = f + u  # 1/v = 1/f + 1/u  =>  v = (f*u)/(f+u)
        if abs(denom) < 1e-6:
            result["infinite"] = True
            return result
        v = (f * u) / denom
        m = v / u
    else:  # mirror
        denom = u - f  # 1/v = 1/f - 1/u => v = f*u/(u-f)
        if abs(denom) < 1e-6:
            result["infinite"] = True
            return result
        v = (f * u) / denom
        m = -v / u

    result["v"] = v
    result["m"] = m
    if is_lens:
        result["real"] = v > 0
    else:
        result["real"] = v < 0
    result["erect"] = m > 0
    return result


def power_of_lens(f_cm):
    """Power in Diopters. f given in cm -> convert to metres."""
    f_m = f_cm / 100.0
    if abs(f_m) < 1e-9:
        return float("inf")
    return 1.0 / f_m


# ----------------------------------------------------------------------
# RAY GEOMETRY HELPERS
# ----------------------------------------------------------------------
def _between(a, lo_pt, hi_pt):
    lo, hi = min(lo_pt, hi_pt), max(lo_pt, hi_pt)
    return lo <= a <= hi


def lens_rays(x_obj, h_obj, f, xmax):
    """Three standard construction rays for a thin lens at x=0."""
    segs = []  # (xs, ys, color, linestyle)
    c0, c1, c2 = RAY_COLORS

    # Ray 1: parallel to axis -> refracts through image-side focus (f,0)
    segs.append(([x_obj, 0], [h_obj, h_obj], c0, '-'))
    if abs(f) > 1e-9:
        slope = (0 - h_obj) / (f - 0)
        y_far = h_obj + slope * (xmax - 0)
        segs.append(([0, xmax], [h_obj, y_far], c0, '-'))
        if f < 0:  # diverging (concave lens) -> dashed virtual back-extension
            segs.append(([0, f], [h_obj, 0], c0, '--'))

    # Ray 2: through optical centre, undeviated
    if abs(x_obj) > 1e-9:
        slope2 = h_obj / x_obj
        y_far2 = slope2 * xmax
        segs.append(([x_obj, xmax], [h_obj, y_far2], c1, '-'))

    # Ray 3: aimed at / through object-side focus F = (-f, 0) -> emerges parallel
    Ff = -f
    if abs(Ff - x_obj) > 1e-9:
        slope3 = (0 - h_obj) / (Ff - x_obj)
        h_lens = h_obj + slope3 * (0 - x_obj)
        if _between(Ff, x_obj, 0):
            segs.append(([x_obj, Ff], [h_obj, 0], c2, '-'))
            segs.append(([Ff, 0], [0, h_lens], c2, '-'))
        else:
            segs.append(([x_obj, 0], [h_obj, h_lens], c2, '-'))
            segs.append(([x_obj, Ff], [h_obj, 0], c2, '--'))
        segs.append(([0, xmax], [h_lens, h_lens], c2, '-'))

    return segs


def mirror_rays(x_obj, h_obj, f, xmin):
    """Three standard construction rays for a spherical mirror at x=0."""
    segs = []
    c0, c1, c2 = RAY_COLORS
    C = 2 * f

    # Ray 1: parallel to axis -> reflects through focus F=(f,0)
    segs.append(([x_obj, 0], [h_obj, h_obj], c0, '-'))
    if abs(f) > 1e-9:
        slope = (0 - h_obj) / (f - 0)
        y_far = h_obj + slope * (xmin - 0)
        segs.append(([0, xmin], [h_obj, y_far], c0, '-'))
        if f > 0:  # convex mirror -> virtual, dashed extension behind mirror
            segs.append(([0, f], [h_obj, 0], c0, '--'))

    # Ray 2: through centre of curvature C -> retro-reflects along same line
    if abs(C - x_obj) > 1e-9:
        slope2 = (0 - h_obj) / (C - x_obj)
        h_hit = h_obj + slope2 * (0 - x_obj)
        segs.append(([x_obj, 0], [h_obj, h_hit], c1, '-'))
        if not _between(C, x_obj, 0):
            segs.append(([x_obj, C], [h_obj, 0], c1, '--'))
        if abs(C) > 1e-9:
            slope2c = (0 - h_hit) / (C - 0)
            y_far2 = h_hit + slope2c * (xmin - 0)
            segs.append(([0, xmin], [h_hit, y_far2], c1, '-'))

    # Ray 3: aimed at / through focus F=(f,0) -> reflects parallel to axis
    if abs(f - x_obj) > 1e-9:
        slope3 = (0 - h_obj) / (f - x_obj)
        h_hit3 = h_obj + slope3 * (0 - x_obj)
        if _between(f, x_obj, 0):
            segs.append(([x_obj, f], [h_obj, 0], c2, '-'))
            segs.append(([f, 0], [0, h_hit3], c2, '-'))
        else:
            segs.append(([x_obj, 0], [h_obj, h_hit3], c2, '-'))
            segs.append(([x_obj, f], [h_obj, 0], c2, '--'))
        segs.append(([0, xmin], [h_hit3, h_hit3], c2, '-'))

    return segs


# ----------------------------------------------------------------------
# ELEMENT SHAPE HELPERS  (real lens / mirror geometry, not glyphs)
# ----------------------------------------------------------------------
def build_lens_outline(convex, H, n=100):
    """Closed polygon outline for a biconvex or biconcave lens of half-height H."""
    y = np.linspace(-H, H, n)
    t = np.clip(1.0 - (y / H) ** 2, 0.0, 1.0)
    if convex:
        bulge = H * 0.17
        x_right = bulge * np.sqrt(t)
        x_left = -bulge * np.sqrt(t)
    else:
        edge_half = H * 0.13
        center_half = H * 0.03
        dip = edge_half - center_half
        x_right = center_half + dip * (1.0 - t)
        x_left = -x_right
    xs = np.concatenate([x_right, x_left[::-1]])
    ys = np.concatenate([y, y[::-1]])
    return xs, ys


def build_mirror_curve(f, H, n=100):
    """Sampled points along the true circular arc of a spherical mirror
    of half-aperture H and signed focal length f (mirror pole at x=0)."""
    R = abs(2 * f)
    C_x = 2 * f
    y = np.linspace(-H, H, n)
    y_c = np.clip(y, -R * 0.999, R * 0.999)
    x = C_x - np.sign(C_x) * np.sqrt(np.maximum(R ** 2 - y_c ** 2, 0.0))
    return x, y


def nice_grid_step(span):
    """Pick a 'nice' (1/2/5 x 10^n) grid step so a span shows ~10 divisions."""
    if span <= 0:
        return 1.0
    raw = span / 10.0
    magnitude = 10 ** np.floor(np.log10(raw))
    residual = raw / magnitude
    if residual < 1.5:
        step = 1
    elif residual < 3.5:
        step = 2
    elif residual < 7.5:
        step = 5
    else:
        step = 10
    return float(step * magnitude)


def shade_hex(hex_color, factor):
    """Lighten a #rrggbb color toward white by `factor` (0..1)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f'#{r:02x}{g:02x}{b:02x}'


def _round_rect_points(x1, y1, x2, y2, r):
    r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
    return [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]


# ----------------------------------------------------------------------
# MODERN ROUNDED BUTTON (canvas based, since ttk can't do true rounded corners)
# ----------------------------------------------------------------------
class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, width=110, height=32,
                 radius=10, bg=PANEL_BG, fg="#0d1117", fill=ACCENT,
                 hover=None, font=("Segoe UI", 10, "bold")):
        super().__init__(parent, width=width, height=height, bg=bg,
                          highlightthickness=0, cursor="hand2")
        self.command = command
        self.fill = fill
        self.hover_fill = hover or shade_hex(fill, 0.18)
        self.radius = radius
        self.width_ = width
        self.height_ = height
        self.text = text
        self.font = font
        self.fg = fg
        self._enabled = True
        self._render(self.fill)
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", lambda e: self._enabled and self._render(self.hover_fill))
        self.bind("<Leave>", lambda e: self._enabled and self._render(self.fill))

    def _round_rect(self, x1, y1, x2, y2, r, **kw):
        points = [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
                  x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
                  x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]
        return self.create_polygon(points, smooth=True, **kw)

    def _render(self, color):
        self.delete("all")
        self._round_rect(1, 1, self.width_ - 1, self.height_ - 1, self.radius,
                          fill=color, outline=color)
        self.create_text(self.width_ / 2, self.height_ / 2, text=self.text,
                          fill=self.fg, font=self.font)

    def _on_click(self, event):
        if self._enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self._enabled = enabled
        self.configure(cursor="hand2" if enabled else "arrow")
        self._render(self.fill if enabled else shade_hex(BORDER, 0.05))


class ToggleSwitch(tk.Canvas):
    """A modern pill-shaped on/off toggle bound to a tk.BooleanVar.
    Used in place of the old-fashioned Checkbutton."""

    def __init__(self, master, variable, command=None, width=46, height=24,
                 bg=None, on_color=ACCENT, off_color=TOGGLE_OFF,
                 knob_color="#ffffff"):
        parent_bg = bg if bg is not None else PANEL_BG
        super().__init__(master, width=width, height=height, bg=parent_bg,
                          highlightthickness=0, bd=0, cursor="hand2")
        self.variable = variable
        self.command = command
        self.w, self.h = width, height
        self.on_color = on_color
        self.off_color = off_color
        self.pad = 3
        self._track = self.create_polygon(
            _round_rect_points(0, 0, width, height, height / 2),
            smooth=True, fill=off_color, outline="")
        r = (height - 2 * self.pad) / 2
        self._knob = self.create_oval(self.pad, self.pad,
                                       self.pad + 2 * r, height - self.pad,
                                       fill=knob_color, outline="")
        self.bind("<Button-1>", self._toggle)
        self._sync()

    def _toggle(self, _e=None):
        self.variable.set(not self.variable.get())
        self._sync()
        if self.command:
            self.command()

    def _sync(self):
        r = (self.h - 2 * self.pad) / 2
        if self.variable.get():
            x0 = self.w - self.pad - 2 * r
            self.itemconfigure(self._track, fill=self.on_color)
        else:
            x0 = self.pad
            self.itemconfigure(self._track, fill=self.off_color)
        self.coords(self._knob, x0, self.pad, x0 + 2 * r, self.h - self.pad)


# ----------------------------------------------------------------------
# MAIN APPLICATION
# ----------------------------------------------------------------------
class OpticsLabApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optics Lab — Image vs Object (Lenses & Mirrors)")
        self.root.geometry("1360x860")
        self.root.configure(bg=BG)

        # ---- scale / range constants ----
        self.BASE_F = 15.0
        self.BASE_H = 20.0
        self.H_MIN = 12.0
        self.H_MAX = 34.0
        self.F_MAG_DEFAULT = 15.0
        self.F_MAG_MIN = 3.0
        self.F_MAG_MAX = 60.0
        self.P_MIN = 100.0 / self.F_MAG_MAX
        self.P_MAX = 100.0 / self.F_MAG_MIN
        self.U_MIN = 1.0
        self.U_MAX = 300.0
        self.AXIS_LIMIT_DEFAULT = 48.0
        self.AXIS_LIMIT_MIN = 12.0
        self.AXIS_LIMIT_MAX = 130.0

        # ---- state ----
        self.element_type = tk.StringVar(value="Convex Lens")
        self.preset_var = tk.StringVar(value="Between F and 2F/C")
        self.show_rays = tk.BooleanVar(value=True)
        self.f_mag = self.F_MAG_DEFAULT
        self.x_obj = -1.5 * self.f_mag
        self.axis_limit = self.AXIS_LIMIT_DEFAULT
        self.at_infinity = False
        self.dragging = False
        self.H = self.BASE_H
        self.h_obj = self.H * 0.5

        self.entry_vars = {}
        self.entry_widgets = {}
        self.error_labels = {}
        self.value_labels = {}

        self._recompute_scale()
        self._setup_style()
        self._build_layout()
        self._update_zoom_label()
        self._draw()

    # ------------------------------------------------------------------
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=PANEL_BG)
        style.configure("TLabel", background=PANEL_BG, foreground=TEXT_COLOR,
                         font=("Segoe UI", 10))
        style.configure("Header.TLabel", background=PANEL_BG, foreground=ACCENT,
                         font=("Segoe UI", 12, "bold"))
        style.configure("Value.TLabel", background=PANEL_BG, foreground=TEXT_COLOR,
                         font=("Consolas", 12, "bold"))
        style.configure("Error.TLabel", background=PANEL_BG, foreground=ERROR_COLOR,
                         font=("Segoe UI", 8, "italic"))
        style.configure("Nature.TLabel", background=PANEL_BG, foreground="#7ee787",
                         font=("Segoe UI", 10, "italic"))

        # Modern combobox
        style.configure("TCombobox", fieldbackground=FIELD_BG, background=FIELD_BG,
                         foreground=TEXT_COLOR, arrowcolor=ACCENT, bordercolor=BORDER,
                         lightcolor=FIELD_BG, darkcolor=FIELD_BG, padding=6,
                         relief="flat", borderwidth=1)
        style.map("TCombobox",
                  fieldbackground=[("readonly", FIELD_BG), ("focus", FIELD_BG)],
                  foreground=[("readonly", TEXT_COLOR)],
                  bordercolor=[("focus", ACCENT), ("!focus", BORDER)],
                  arrowcolor=[("active", ACCENT), ("!active", MUTED_TEXT)])

        # Modern entry
        style.configure("Modern.TEntry", fieldbackground=FIELD_BG, foreground=TEXT_COLOR,
                         insertcolor=TEXT_COLOR, bordercolor=BORDER, relief="flat",
                         padding=5, borderwidth=1)
        style.map("Modern.TEntry",
                  fieldbackground=[("disabled", "#161a20"), ("focus", FIELD_BG)],
                  foreground=[("disabled", MUTED_TEXT)],
                  bordercolor=[("focus", ACCENT), ("!focus", BORDER)])

        style.configure("TCheckbutton", background=PANEL_BG, foreground=TEXT_COLOR,
                         font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", PANEL_BG)],
                  foreground=[("active", ACCENT)])

        # Dark dropdown listboxes (tk classic widget used by ttk combobox popdown)
        self.root.option_add("*TCombobox*Listbox.background", FIELD_BG)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT_COLOR)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#0d1117")
        self.root.option_add("*TCombobox*Listbox.font", ("Segoe UI", 10))

    # ------------------------------------------------------------------
    def _build_layout(self):
        # ---- top control bar ----
        top = tk.Frame(self.root, bg=PANEL_BG, pady=10, padx=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Optical Element:", style="Header.TLabel").grid(
            row=0, column=0, padx=(0, 6), sticky="w")
        elem_combo = ttk.Combobox(top, textvariable=self.element_type,
                                   values=ELEMENT_TYPES, state="readonly", width=15)
        elem_combo.grid(row=0, column=1, padx=(0, 18))
        elem_combo.bind("<<ComboboxSelected>>", self._on_element_change)

        ttk.Label(top, text="Object Position:", style="Header.TLabel").grid(
            row=0, column=2, padx=(0, 6), sticky="w")
        preset_combo = ttk.Combobox(top, textvariable=self.preset_var,
                                     values=PRESETS, state="readonly", width=19)
        preset_combo.grid(row=0, column=3, padx=(0, 18))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_change)

        ray_row = tk.Frame(top, bg=PANEL_BG)
        ray_row.grid(row=0, column=4, padx=(0, 18), sticky="w")
        ttk.Label(ray_row, text="Full Ray Diagram", style="TLabel").pack(
            side=tk.LEFT, padx=(0, 8))
        self.ray_toggle = ToggleSwitch(ray_row, variable=self.show_rays,
                                        command=self._draw)
        self.ray_toggle.pack(side=tk.LEFT)

        # zoom controls
        zoom_frame = tk.Frame(top, bg=PANEL_BG)
        zoom_frame.grid(row=0, column=5, padx=(0, 18))
        ttk.Label(zoom_frame, text="Zoom", style="TLabel", foreground=MUTED_TEXT).pack(
            side=tk.LEFT, padx=(0, 6))
        RoundedButton(zoom_frame, "–", command=lambda: self._zoom(1 / 0.88),
                      width=30, height=28, radius=8, fill="#21262d",
                      fg=TEXT_COLOR).pack(side=tk.LEFT, padx=1)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", style="TLabel", width=5,
                                     anchor="center")
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        RoundedButton(zoom_frame, "+", command=lambda: self._zoom(0.88),
                      width=30, height=28, radius=8, fill="#21262d",
                      fg=TEXT_COLOR).pack(side=tk.LEFT, padx=1)

        RoundedButton(top, "Reset", command=self._reset, width=90, height=32,
                      fill=ACCENT, fg="#0d1117").grid(row=0, column=6, padx=(0, 8))
        RoundedButton(top, "ⓘ About", command=self._show_about, width=100, height=32,
                      fill="#21262d", fg=TEXT_COLOR, hover=shade_hex("#21262d", 0.25)
                      ).grid(row=0, column=7)

        hint = ttk.Label(top, text=("Tip: drag the cyan arrow  •  type exact values in "
                                     "the panel  •  Ctrl + Scroll to zoom  •  try "
                                     "At Infinity"),
                          style="TLabel", foreground=MUTED_TEXT)
        hint.grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))

        # ---- body: canvas (left) + info panel (right) ----
        body = tk.Frame(self.root, bg=BG)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9.5, 6.8), dpi=100, facecolor=BG)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                                          padx=(10, 5), pady=10)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self._bind_zoom()

        # ---- right info panel ----
        panel = tk.Frame(body, bg=PANEL_BG, width=300)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 10), pady=10)
        panel.pack_propagate(False)

        canvas_scroll = tk.Canvas(panel, bg=PANEL_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(panel, orient="vertical", command=canvas_scroll.yview)
        inner = tk.Frame(canvas_scroll, bg=PANEL_BG)
        inner.bind("<Configure>", lambda e: canvas_scroll.configure(
            scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=inner, anchor="nw", width=280)
        canvas_scroll.configure(yscrollcommand=vsb.set)
        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(inner, text="MEASUREMENTS", style="Header.TLabel").pack(
            anchor="w", padx=14, pady=(14, 2))
        ttk.Label(inner, text="Type a value and press Enter to apply it.",
                  style="TLabel", foreground=MUTED_TEXT, font=("Segoe UI", 8)).pack(
            anchor="w", padx=14, pady=(0, 8))

        self._build_editable_row(inner, "f", "Focal length |f|  (3–60 cm)", "cm",
                                  self._commit_f)
        self._build_editable_row(inner, "u", "Object distance u  (signed, cm)", "cm",
                                  self._commit_u)
        self._build_editable_row(inner, "v", "Image distance v  (signed, cm)", "cm",
                                  self._commit_v)
        self._build_editable_row(inner, "P", "Power P  (lens only, D)", "D",
                                  self._commit_P)

        # magnification stays read-only (purely derived, not requested editable)
        row = tk.Frame(inner, bg=PANEL_BG)
        row.pack(anchor="w", fill="x", padx=14, pady=3)
        ttk.Label(row, text="Magnification (m)", style="TLabel",
                  foreground=MUTED_TEXT).pack(anchor="w")
        lbl = ttk.Label(row, text="--", style="Value.TLabel")
        lbl.pack(anchor="w")
        self.value_labels["m"] = lbl

        ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=14, pady=10)

        ttk.Label(inner, text="IMAGE NATURE", style="Header.TLabel").pack(
            anchor="w", padx=14, pady=(0, 6))
        self.nature_label = ttk.Label(inner, text="--", style="Nature.TLabel",
                                       wraplength=250, justify="left")
        self.nature_label.pack(anchor="w", padx=14)

        ttk.Separator(inner, orient="horizontal").pack(fill="x", padx=14, pady=10)

        ttk.Label(inner, text="LEGEND", style="Header.TLabel").pack(
            anchor="w", padx=14, pady=(0, 6))
        legend_items = [
            (OBJECT_COLOR, "Object"),
            (IMAGE_REAL_COLOR, "Real image (solid)"),
            (IMAGE_VIRTUAL_COLOR, "Virtual image (dashed)"),
            (FOCUS_COLOR, "Focus (F)"),
            (CENTER_COLOR, "Centre of curvature (C) / 2F"),
            (RAY_COLORS[0], "Ray 1 (parallel  focus)"),
            (RAY_COLORS[1], "Ray 2 (through centre)"),
            (RAY_COLORS[2], "Ray 3 (through focus  parallel)"),
        ]
        for color, text in legend_items:
            row = tk.Frame(inner, bg=PANEL_BG)
            row.pack(anchor="w", fill="x", padx=14, pady=2)
            swatch = tk.Canvas(row, width=16, height=10, bg=PANEL_BG,
                                highlightthickness=0)
            swatch.create_rectangle(0, 0, 16, 10, fill=color, outline=color)
            swatch.pack(side=tk.LEFT, padx=(0, 8))
            ttk.Label(row, text=text, style="TLabel").pack(side=tk.LEFT)

        ttk.Label(inner, text="", style="TLabel").pack(pady=6)  # bottom breathing room

    def _build_editable_row(self, parent, key, desc, unit, handler):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(anchor="w", fill="x", padx=14, pady=4)
        ttk.Label(row, text=desc, style="TLabel", foreground=MUTED_TEXT).pack(anchor="w")
        entry_row = tk.Frame(row, bg=PANEL_BG)
        entry_row.pack(anchor="w", pady=(3, 0))
        var = tk.StringVar()
        entry = ttk.Entry(entry_row, textvariable=var, width=11, style="Modern.TEntry",
                           font=("Consolas", 11))
        entry.pack(side=tk.LEFT)
        entry.bind("<Return>", handler)
        entry.bind("<FocusOut>", handler)
        ttk.Label(entry_row, text=f" {unit}", style="TLabel",
                  foreground=MUTED_TEXT).pack(side=tk.LEFT, padx=(6, 0))
        err = ttk.Label(row, text="", style="Error.TLabel")
        err.pack(anchor="w")
        self.entry_vars[key] = var
        self.entry_widgets[key] = entry
        self.error_labels[key] = err

    def _show_field_error(self, key, msg):
        self.error_labels[key].configure(text=msg)

    def _clear_field_error(self, key):
        self.error_labels[key].configure(text="")

    # ------------------------------------------------------------------
    # ABOUT DIALOG
    # ------------------------------------------------------------------
    def _show_about(self):
        win = tk.Toplevel(self.root)
        win.title("About Optics Lab")
        win.configure(bg=PANEL_BG)
        win.resizable(False, False)
        win.transient(self.root)

        self.root.update_idletasks()
        w, h = 600, 300
        x = self.root.winfo_x() + self.root.winfo_width() // 2 - w // 2
        y = self.root.winfo_y() + self.root.winfo_height() // 2 - h // 2
        win.geometry(f"{w}x{h}+{max(x,0)}+{max(y,0)}")

        container = tk.Frame(win, bg=PANEL_BG, padx=24, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(container, text="Optics Lab", style="Header.TLabel",
                  font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(container, text="Image vs Object Formation for Spherical Lenses "
                                   "& Mirrors [Helpfull for CBSE Class 10 Students]", style="TLabel", foreground=MUTED_TEXT,
                  wraplength=290, justify="left").pack(anchor="w", pady=(3, 14))

        ttk.Separator(container, orient="horizontal").pack(fill="x", pady=(0, 12))

        row1 = tk.Frame(container, bg=PANEL_BG)
        row1.pack(anchor="w", fill="x", pady=2)
        ttk.Label(row1, text="Developed by:", style="TLabel",
                  foreground=MUTED_TEXT).pack(side=tk.LEFT)
        ttk.Label(row1, text=f" {APP_AUTHOR}", style="Value.TLabel").pack(side=tk.LEFT)

        row2 = tk.Frame(container, bg=PANEL_BG)
        row2.pack(anchor="w", fill="x", pady=2)
        ttk.Label(row2, text="Sent feedback or suggetions at Email:", style="TLabel",
                  foreground=MUTED_TEXT).pack(side=tk.LEFT)
        ttk.Label(row2, text=f" {APP_EMAIL}", style="Value.TLabel").pack(side=tk.LEFT)

        RoundedButton(container, "Close", command=win.destroy, width=100, height=32,
                      fill=ACCENT, fg="#0d1117").pack(pady=(18, 0))

        win.bind("<Escape>", lambda e: win.destroy())
        win.focus_set()
        win.grab_set()

    # ------------------------------------------------------------------
    # SCALE / ZOOM
    # ------------------------------------------------------------------
    def _recompute_scale(self):
        H = self.BASE_H * (self.f_mag / self.BASE_F)
        self.H = max(self.H_MIN, min(self.H_MAX, H))
        self.h_obj = self.H * 0.5

    def _autofit_axis(self):
        etype = self.element_type.get()
        f = signed_focal_length(etype, self.f_mag)
        candidates = [abs(self.x_obj), self.f_mag * 3.2]
        if not self.at_infinity:
            result = compute_image(etype, self.x_obj, f)
            if not result["infinite"] and result["v"] is not None:
                candidates.append(abs(result["v"]))
        needed = max(candidates) * 1.18
        needed = max(24.0, min(self.AXIS_LIMIT_MAX, needed))
        if needed > self.axis_limit:
            self.axis_limit = needed
            self._update_zoom_label()

    def _bind_zoom(self):
        widget = self.canvas.get_tk_widget()
        widget.bind("<MouseWheel>", self._on_wheel)          # Windows / macOS
        widget.bind("<Button-4>", lambda e: self._on_wheel(e, linux_dir=1))   # Linux up
        widget.bind("<Button-5>", lambda e: self._on_wheel(e, linux_dir=-1))  # Linux down

    def _on_wheel(self, event, linux_dir=None):
        ctrl_held = bool(event.state & 0x0004)
        if not ctrl_held:
            return None
        if linux_dir is not None:
            factor = 0.88 if linux_dir > 0 else (1 / 0.88)
        else:
            factor = 0.88 if event.delta > 0 else (1 / 0.88)
        self._zoom(factor)
        return "break"

    def _zoom(self, factor):
        new_limit = self.axis_limit * factor
        new_limit = max(self.AXIS_LIMIT_MIN, min(self.AXIS_LIMIT_MAX, new_limit))
        self.axis_limit = new_limit
        self._update_zoom_label()
        self._draw()

    def _update_zoom_label(self):
        pct = (self.AXIS_LIMIT_DEFAULT / self.axis_limit) * 100.0
        self.zoom_label.configure(text=f"{pct:.0f}%")

    # ------------------------------------------------------------------
    # MANUAL VALUE ENTRY HANDLERS
    # ------------------------------------------------------------------
    def _commit_f(self, event=None):
        raw = self.entry_vars["f"].get().strip()
        try:
            val = float(raw)
        except ValueError:
            self._show_field_error("f", "Enter a number")
            return "break"
        if not (self.F_MAG_MIN <= val <= self.F_MAG_MAX):
            self._show_field_error("f", f"Range {self.F_MAG_MIN:.0f}{self.F_MAG_MAX:.0f} cm")
            return "break"
        self._clear_field_error("f")
        self.f_mag = val
        self._recompute_scale()
        self._autofit_axis()
        self._draw()
        return "break"

    def _commit_P(self, event=None):
        if str(self.entry_widgets["P"].cget("state")) == "disabled":
            return "break"
        raw = self.entry_vars["P"].get().strip()
        if raw == "":
            return "break"
        try:
            val = float(raw)
        except ValueError:
            self._show_field_error("P", "Enter a number")
            return "break"
        if val <= 0:
            self._show_field_error("P", "Must be > 0")
            return "break"
        f_mag = 100.0 / val
        if not (self.F_MAG_MIN <= f_mag <= self.F_MAG_MAX):
            self._show_field_error("P", f"Range {self.P_MIN:.2f}{self.P_MAX:.2f} D")
            return "break"
        self._clear_field_error("P")
        self.f_mag = f_mag
        self._recompute_scale()
        self._autofit_axis()
        self._draw()
        return "break"

    def _commit_u(self, event=None):
        raw = self.entry_vars["u"].get().strip()
        if raw in ("\u2212\u221e", "-\u221e", "-inf"):
            return "break"  # placeholder from "At Infinity" mode, not a real edit
        try:
            val = float(raw)
        except ValueError:
            self._show_field_error("u", "Enter a number")
            return "break"
        mag = abs(val)
        if not (self.U_MIN <= mag <= self.U_MAX):
            self._show_field_error("u", f"Range {self.U_MIN:.0f}{self.U_MAX:.0f} cm")
            return "break"
        self._clear_field_error("u")
        self.x_obj = -mag
        self.at_infinity = False
        self.preset_var.set("Custom (drag me)")
        self._autofit_axis()
        self._draw()
        return "break"

    def _commit_v(self, event=None):
        raw = self.entry_vars["v"].get().strip()
        if raw in ("\u221e", "inf"):
            return "break"  # placeholder for "image at infinity", not a real edit
        try:
            val = float(raw)
        except ValueError:
            self._show_field_error("v", "Enter a number")
            return "break"
        etype = self.element_type.get()
        is_lens = "Lens" in etype
        f = signed_focal_length(etype, self.f_mag)
        denom = (f - val) if is_lens else (val - f)
        if abs(denom) < 1e-6:
            self._show_field_error("v", "No finite object for this v")
            return "break"
        u = (val * f) / denom
        if u > -self.U_MIN:
            self._show_field_error("v", "Not achievable with a real object")
            return "break"
        if abs(u) > self.U_MAX:
            self._show_field_error("v", "Resulting u out of range")
            return "break"
        self._clear_field_error("v")
        self.x_obj = u
        self.at_infinity = False
        self.preset_var.set("Custom (drag me)")
        self._autofit_axis()
        self._draw()
        return "break"

    # ------------------------------------------------------------------
    # EVENT HANDLERS
    # ------------------------------------------------------------------
    def _on_element_change(self, event=None):
        is_lens = "Lens" in self.element_type.get()
        self.entry_widgets["P"].configure(state="normal" if is_lens else "disabled")
        if not is_lens:
            self.entry_vars["P"].set("")
            self._clear_field_error("P")
        self._autofit_axis()
        self._draw()

    def _on_preset_change(self, event=None):
        preset = self.preset_var.get()
        if preset == "At Infinity":
            self.at_infinity = True
            self._autofit_axis()
            self._draw()
            return
        self.at_infinity = False
        x = preset_to_x(preset, self.f_mag)
        if x is not None:
            self.x_obj = x
        self._autofit_axis()
        self._draw()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.at_infinity:
            new_x = max(-self.axis_limit + 2, min(-self.U_MIN, event.xdata))
            self.at_infinity = False
            self.x_obj = new_x
            self.preset_var.set("Custom (drag me)")
            self.dragging = True
            self._draw()
            return
        tol_x = self.axis_limit * 0.035
        tol_y = max(self.h_obj * 0.35, 1.5)
        if abs(event.xdata - self.x_obj) < tol_x and -tol_y < event.ydata < self.h_obj + tol_y:
            self.dragging = True

    def _on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax or event.xdata is None:
            return
        new_x = event.xdata
        new_x = max(-self.axis_limit + 2, min(-self.U_MIN, new_x))
        # avoid the singular points (object exactly at a focal-type distance
        # that sends the image to infinity) by nudging slightly
        f = signed_focal_length(self.element_type.get(), self.f_mag)
        singular = -f if "Lens" in self.element_type.get() else f
        if abs(new_x - singular) < 0.3:
            new_x = singular - 0.3 if new_x < singular else singular + 0.3
        self.x_obj = new_x
        self.preset_var.set("Custom (drag me)")
        self._draw()

    def _on_release(self, event):
        self.dragging = False

    def _reset(self):
        self.element_type.set("Convex Lens")
        self.preset_var.set("Between F and 2F/C")
        self.f_mag = self.F_MAG_DEFAULT
        self.x_obj = -1.5 * self.f_mag
        self.show_rays.set(True)
        self.axis_limit = self.AXIS_LIMIT_DEFAULT
        self.at_infinity = False
        self._recompute_scale()
        self.entry_widgets["P"].configure(state="normal")
        for k in self.error_labels:
            self._clear_field_error(k)
        self._update_zoom_label()
        self._draw()

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------
    def _draw(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor(BG)

        etype = self.element_type.get()
        is_lens = "Lens" in etype
        f = signed_focal_length(etype, self.f_mag)

        limit = self.axis_limit
        Y = self.H + 9
        major_step = nice_grid_step(2 * limit)
        minor_step = major_step / 5.0

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-Y, Y)
        ax.set_xticks(np.arange(-limit - major_step, limit + major_step, major_step))
        ax.set_xticks(np.arange(-limit - minor_step, limit + minor_step, minor_step),
                       minor=True)
        ax.grid(which="major", color=GRID_MAJOR, linewidth=0.7, alpha=0.9)
        ax.grid(which="minor", color=GRID_MINOR, linewidth=0.4, alpha=0.5)
        ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_MAJOR)
        ax.axhline(0, color=AXIS_LINE, linewidth=1.4, zorder=2)
        ax.set_xlabel("Distance from optical centre / pole  (cm)", color=MUTED_TEXT,
                       fontsize=9)
        ax.set_title(f"{etype} — Ray Diagram" if self.show_rays.get() else
                     f"{etype} — Object & Image", color=TEXT_COLOR, fontsize=13,
                     fontweight="bold", pad=12)

        # ---- draw the optical element at x = 0 ----
        self._draw_element(ax, etype, f)

        # ---- focus / centre markers ----
        self._draw_markers(ax, etype, f)

        # ---- object/image (or infinity case) ----
        if self.at_infinity:
            self._draw_infinity_case(ax, etype, f, is_lens)
        else:
            self._draw_finite_case(ax, etype, f, is_lens)

        # single, unconditional redraw call — this is what keeps the
        # canvas in sync with every state change (drag, dropdown, typed value)
        self.canvas.draw_idle()

    def _draw_finite_case(self, ax, etype, f, is_lens):
        u = self.x_obj
        h_obj = self.h_obj
        H = self.H
        result = compute_image(etype, u, f)

        self._draw_arrow(ax, u, h_obj, OBJECT_COLOR, "O")

        if result["infinite"]:
            ax.text(0, H, "Image forms at INFINITY\n(object at focal point)",
                    color=IMAGE_VIRTUAL_COLOR, fontsize=11, ha="center",
                    fontweight="bold")
            self._draw_dimension(ax, 0, u, -(H + 2), f"u = {u:+.1f} cm", DIM_U_COLOR)
            self._draw_dimension(ax, 0, f, H + 4, f"f = {f:+.1f} cm", DIM_F_COLOR)
            self._update_info_panel(etype, u, None, f, None, result)
            return

        v = result["v"]
        m = result["m"]
        h_img = m * h_obj
        img_color = IMAGE_REAL_COLOR if result["real"] else IMAGE_VIRTUAL_COLOR

        if self.show_rays.get():
            xmax = self.axis_limit - 1
            xmin = -self.axis_limit + 1
            segs = (lens_rays(u, h_obj, f, xmax) if is_lens
                    else mirror_rays(u, h_obj, f, xmin))
            for xs, ys, color, ls in segs:
                ax.plot(xs, ys, color=color, linewidth=1.5, linestyle=ls,
                        alpha=0.95, zorder=3)

        self._draw_arrow(ax, v, h_img, img_color, "I", dashed=not result["real"])

        self._draw_dimension(ax, 0, u, -(H + 2), f"u = {u:+.1f} cm", DIM_U_COLOR)
        self._draw_dimension(ax, 0, v, -(H + 6), f"v = {v:+.1f} cm", DIM_V_COLOR)
        self._draw_dimension(ax, 0, f, H + 4, f"f = {f:+.1f} cm", DIM_F_COLOR)

        self._update_info_panel(etype, u, v, f, m, result)

    def _draw_infinity_case(self, ax, etype, f, is_lens):
        H = self.H
        limit = self.axis_limit
        heights = [H * 0.8, H * 0.4, -H * 0.4, -H * 0.8]
        v = f
        real = (v > 0) if is_lens else (v < 0)
        img_color = IMAGE_REAL_COLOR if real else IMAGE_VIRTUAL_COLOR

        if self.show_rays.get():
            xmax = limit - 1
            xmin = -limit + 1
            for h in heights:
                segs = [([xmin, 0], [h, h], '-')]
                if abs(f) > 1e-9:
                    if is_lens:
                        slope = (0 - h) / (f - 0)
                        y_far = h + slope * (xmax - 0)
                        segs.append(([0, xmax], [h, y_far], '-'))
                        if f < 0:
                            segs.append(([0, f], [h, 0], '--'))
                    else:
                        slope = (0 - h) / (f - 0)
                        y_far = h + slope * (xmin - 0)
                        segs.append(([0, xmin], [h, y_far], '-'))
                        if f > 0:
                            segs.append(([0, f], [h, 0], '--'))
                for xs, ys, ls in segs:
                    ax.plot(xs, ys, color=INFINITY_RAY_COLOR, linewidth=1.5,
                            linestyle=ls, alpha=0.9, zorder=3)
            ax.annotate("Parallel rays from object at infinity",
                        xy=(xmin + 2, heights[0] + 1.5), color=INFINITY_RAY_COLOR,
                        fontsize=8.5, fontweight="bold")

        ax.plot(v, 0, "o", color=img_color, markersize=10, zorder=6,
                markeredgecolor=TEXT_COLOR, markeredgewidth=1.2)
        ax.annotate("I", (v, 0), textcoords="offset points", xytext=(8, 8),
                    color=img_color, fontsize=11, fontweight="bold", zorder=6)

        note = ("Object at Infinity  \u2192  Real point image\nforms at the focus."
                if real else
                "Object at Infinity  \u2192  Virtual point image\nforms at the focus.")
        ax.text(0, H + 4, note, color=img_color, fontsize=9.5, ha="center",
                fontweight="bold")

        self._draw_dimension(ax, 0, v, -(H + 2), f"v = f = {v:+.1f} cm", DIM_F_COLOR)

        ax.annotate("", xy=(-limit + 2, -(H - 2)), xytext=(-limit + 11, -(H - 2)),
                    arrowprops=dict(arrowstyle="-|>", color=DIM_U_COLOR, lw=1.6))
        ax.text(-limit + 6.5, -(H - 4.5), "u \u2192 \u2212\u221e", color=DIM_U_COLOR,
                fontsize=9, fontweight="bold", ha="center")

        self._update_info_panel_infinity(etype, f, v, real)

    def _draw_element(self, ax, etype, f):
        H = self.H
        if "Lens" in etype:
            convex = etype == "Convex Lens"
            xs, ys = build_lens_outline(convex, H)
            ax.fill(xs, ys, color=ELEMENT_LENS_COLOR, alpha=0.20, zorder=3.5)
            ax.plot(xs, ys, color=ELEMENT_LENS_COLOR, linewidth=2.3, zorder=4,
                    solid_capstyle="round")
            ax.text(0, H + 2.4, "Convex Lens" if convex else "Concave Lens",
                    color=ELEMENT_LENS_COLOR, fontsize=9, ha="center",
                    fontweight="bold")
        else:
            concave = etype == "Concave Mirror"
            xs, ys = build_mirror_curve(f, H)
            ax.plot(xs, ys, color=ELEMENT_MIRROR_COLOR, linewidth=3.2, zorder=4,
                    solid_capstyle="round")
            back_off = H * 0.095
            idxs = np.linspace(0, len(xs) - 1, 11).astype(int)
            for i in idxs:
                ax.plot([xs[i], xs[i] + back_off], [ys[i], ys[i]],
                        color=ELEMENT_MIRROR_COLOR, linewidth=1.0, alpha=0.55,
                        zorder=3)
            ax.text(0, H + 2.4, "Concave Mirror" if concave else "Convex Mirror",
                    color=ELEMENT_MIRROR_COLOR, fontsize=9, ha="center",
                    fontweight="bold")

    def _draw_markers(self, ax, etype, f):
        is_lens = "Lens" in etype
        pts = [(f, "F'" if is_lens else "F"), (-f, "F")] if is_lens else [(f, "F")]
        for x, label in pts:
            ax.plot(x, 0, "o", color=FOCUS_COLOR, markersize=6, zorder=5)
            ax.annotate(label, (x, 0), textcoords="offset points", xytext=(0, 10),
                        color=FOCUS_COLOR, fontsize=9, ha="center", fontweight="bold")
        if is_lens:
            for x, label in [(2 * f, "2F'"), (-2 * f, "2F")]:
                ax.plot(x, 0, "o", color=CENTER_COLOR, markersize=6, zorder=5)
                ax.annotate(label, (x, 0), textcoords="offset points", xytext=(0, -14),
                            color=CENTER_COLOR, fontsize=9, ha="center",
                            fontweight="bold")
        else:
            x = 2 * f
            ax.plot(x, 0, "o", color=CENTER_COLOR, markersize=6, zorder=5)
            ax.annotate("C", (x, 0), textcoords="offset points", xytext=(0, -14),
                        color=CENTER_COLOR, fontsize=9, ha="center", fontweight="bold")
        # pole / optical centre
        ax.plot(0, 0, "o", color=TEXT_COLOR, markersize=5, zorder=5)
        ax.annotate("O" if is_lens else "P", (0, 0), textcoords="offset points",
                    xytext=(6, -14), color=TEXT_COLOR, fontsize=9, fontweight="bold")

    def _draw_arrow(self, ax, x, h, color, label, dashed=False):
        if abs(h) < 1e-6:
            h = 0.3 if h >= 0 else -0.3
        ax.annotate("", xy=(x, h), xytext=(x, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                     linewidth=2.6, mutation_scale=20,
                                     linestyle="dashed" if dashed else "solid"),
                    zorder=6)
        ax.annotate(label, (x, h), textcoords="offset points",
                    xytext=(8, 6 if h >= 0 else -14), color=color,
                    fontsize=10, fontweight="bold", zorder=6)

    def _draw_dimension(self, ax, x1, x2, y, text, color):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=1.3),
                    zorder=3)
        mid = (x1 + x2) / 2
        ax.text(mid, y + (1.3 if y >= 0 else -2.6), text, color=color,
                fontsize=8.5, ha="center", fontweight="bold")

    # ------------------------------------------------------------------
    def _update_info_panel(self, etype, u, v, f, m, result):
        is_lens = "Lens" in etype
        self.entry_vars["f"].set(f"{abs(f):.2f}")
        self.entry_vars["u"].set(f"{u:+.2f}")
        self.entry_vars["v"].set("\u221e" if result["infinite"] else f"{v:+.2f}")
        if is_lens:
            P = power_of_lens(f)
            self.entry_vars["P"].set(f"{abs(P):.2f}")
        else:
            self.entry_vars["P"].set("")
        self.value_labels["m"].configure(
            text="--" if result["infinite"] else f"{m:+.3f}")

        if result["infinite"]:
            self.nature_label.configure(
                text="Object is at the focal point.\nRays emerge parallel — "
                     "image forms at infinity.")
            return

        nature_bits = []
        nature_bits.append("Real" if result["real"] else "Virtual")
        nature_bits.append("Inverted" if not result["erect"] else "Erect")
        mag = abs(m)
        if mag > 1.001:
            nature_bits.append("Magnified")
        elif mag < 0.999:
            nature_bits.append("Diminished")
        else:
            nature_bits.append("Same size")
        self.nature_label.configure(text=" • ".join(nature_bits))

    def _update_info_panel_infinity(self, etype, f, v, real):
        is_lens = "Lens" in etype
        self.entry_vars["f"].set(f"{abs(f):.2f}")
        self.entry_vars["u"].set("\u2212\u221e")
        self.entry_vars["v"].set(f"{v:+.2f}")
        if is_lens:
            P = power_of_lens(f)
            self.entry_vars["P"].set(f"{abs(P):.2f}")
        else:
            self.entry_vars["P"].set("")
        self.value_labels["m"].configure(text="\u2248 0 (point)")
        nature = "Real" if real else "Virtual"
        tail = "Rays converge there." if real else "Rays appear to diverge from it."
        self.nature_label.configure(
            text=f"{nature} point image at the principal focus.\n{tail}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = OpticsLabApp(root)
    root.mainloop()
