import argparse
import decimal
import math
import queue
import sys
import threading
import time

import cupy as cp
import numpy as np
import pygame

# Python-side precision is raised automatically as CUDA limb precision rises.
decimal.getcontext().prec = 2000
LN2 = decimal.Decimal(2).ln()


# =============================
# Precision / fixed-point helpers
# =============================

def ensure_decimal_precision(n_limbs, guard_digits=80):
    required_digits = int(math.ceil(n_limbs * 32 * math.log10(2))) + guard_digits
    ctx = decimal.getcontext()
    if ctx.prec < required_digits:
        old = ctx.prec
        ctx.prec = required_digits
        print(f"[PRECISION] Python Decimal precision increased: {old} -> {ctx.prec} digits")
    return ctx.prec


def bits_to_limbs(bits):
    return max(2, int(math.ceil(bits / 32.0)))


def limbs_to_bits(n_limbs):
    return int(n_limbs * 32)


def short_dec(v, digits=18):
    return f"{v:.{digits}E}"


def decimal_to_limbs(val, n_limbs):
    ensure_decimal_precision(n_limbs)
    val_dec = decimal.Decimal(val)
    frac_bits = 32 * (n_limbs - 1)
    scale = decimal.Decimal(1 << frac_bits)
    scaled_val = int(val_dec * scale)

    # Two's complement for negative fixed-point values.
    if scaled_val < 0:
        scaled_val = (1 << (32 * n_limbs)) + scaled_val

    limbs = np.zeros(n_limbs, dtype=np.uint32)
    for i in range(n_limbs - 1, -1, -1):
        limbs[i] = scaled_val & 0xFFFFFFFF
        scaled_val >>= 32
    return limbs


def compile_dynamic_kernel(n_limbs):
    cuda_source = f'''
    #define N_LIMBS {n_limbs}
    extern "C" {{
        struct BigFixed {{ unsigned int limbs[N_LIMBS]; }};

        __device__ BigFixed add(const BigFixed& a, const BigFixed& b) {{
            BigFixed res = {{0}}; unsigned long long carry = 0;
            for (int i = N_LIMBS - 1; i >= 0; --i) {{
                unsigned long long sum = (unsigned long long)a.limbs[i] + b.limbs[i] + carry;
                res.limbs[i] = (unsigned int)sum;
                carry = sum >> 32;
            }}
            return res;
        }}

        __device__ BigFixed negate(const BigFixed& a) {{
            BigFixed res = {{0}}; unsigned long long carry = 1;
            for(int i = N_LIMBS - 1; i >= 0; --i) {{
                unsigned long long sum = (unsigned long long)(~a.limbs[i]) + carry;
                res.limbs[i] = (unsigned int)sum;
                carry = sum >> 32;
            }}
            return res;
        }}

        __device__ BigFixed sub(const BigFixed& a, const BigFixed& b) {{ return add(a, negate(b)); }}

        __device__ BigFixed mul(const BigFixed& a, const BigFixed& b) {{
            bool a_neg = (a.limbs[0] & 0x80000000) != 0;
            bool b_neg = (b.limbs[0] & 0x80000000) != 0;
            BigFixed abs_a = a_neg ? negate(a) : a;
            BigFixed abs_b = b_neg ? negate(b) : b;
            BigFixed res = {{0}}; unsigned int temp[N_LIMBS * 2] = {{0}};

            for (int i = N_LIMBS - 1; i >= 0; --i) {{
                unsigned long long carry = 0;
                for (int j = N_LIMBS - 1; j >= 0; --j) {{
                    int pos = i + j + 1;
                    unsigned long long prod = (unsigned long long)abs_a.limbs[i] * abs_b.limbs[j] + temp[pos] + carry;
                    temp[pos] = (unsigned int)prod;
                    carry = prod >> 32;
                }}
                temp[i] += (unsigned int)carry;
            }}

            // Fixed-point radix alignment: one integer limb, remaining fractional limbs.
            for(int i = 0; i < N_LIMBS; i++) res.limbs[i] = temp[i + 1];
            return (a_neg != b_neg) ? negate(res) : res;
        }}

        __device__ BigFixed mul_uint(const BigFixed& a, unsigned int b) {{
            BigFixed res = {{0}}; unsigned long long carry = 0;
            for(int i = N_LIMBS - 1; i >= 0; --i) {{
                unsigned long long prod = (unsigned long long)a.limbs[i] * b + carry;
                res.limbs[i] = (unsigned int)prod;
                carry = prod >> 32;
            }}
            return res;
        }}

        __device__ BigFixed fixed_uint(unsigned int v) {{
            BigFixed res = {{0}};
            res.limbs[0] = v;
            return res;
        }}

        __device__ BigFixed fixed_quarter() {{
            BigFixed res = {{0}};
            res.limbs[1] = 0x40000000u;
            return res;
        }}

        __device__ BigFixed fixed_sixteenth() {{
            BigFixed res = {{0}};
            res.limbs[1] = 0x10000000u;
            return res;
        }}

        __device__ int cmp_signed(const BigFixed& a, const BigFixed& b) {{
            bool a_neg = (a.limbs[0] & 0x80000000u) != 0;
            bool b_neg = (b.limbs[0] & 0x80000000u) != 0;
            if (a_neg != b_neg) return a_neg ? -1 : 1;
            for (int i = 0; i < N_LIMBS; ++i) {{
                if (a.limbs[i] < b.limbs[i]) return -1;
                if (a.limbs[i] > b.limbs[i]) return 1;
            }}
            return 0;
        }}

        __device__ bool le_signed(const BigFixed& a, const BigFixed& b) {{
            return cmp_signed(a, b) <= 0;
        }}

        __device__ bool escaped_radius4(const BigFixed& zx2, const BigFixed& zy2) {{
            // Only need to know whether zx2+zy2 reached integer part 4.
            // This avoids building a temporary BigFixed sum every iteration.
            unsigned long long carry = 0;
            for (int i = N_LIMBS - 1; i >= 1; --i) {{
                unsigned long long sum = (unsigned long long)zx2.limbs[i] + zy2.limbs[i] + carry;
                carry = sum >> 32;
            }}
            unsigned long long top = (unsigned long long)zx2.limbs[0] + zy2.limbs[0] + carry;
            return top >= 4ull;
        }}

        __device__ bool in_main_cardioid_or_period2_bulb(const BigFixed& cx, const BigFixed& cy, const BigFixed& cy2) {{
            BigFixed one = fixed_uint(1);
            BigFixed quarter = fixed_quarter();
            BigFixed sixteenth = fixed_sixteenth();

            // Period-2 bulb: (x + 1)^2 + y^2 <= 1/16
            BigFixed xp1 = add(cx, one);
            BigFixed bulb2 = add(mul(xp1, xp1), cy2);
            if (le_signed(bulb2, sixteenth)) return true;

            // Main cardioid: q*(q + x - 1/4) <= y^2/4, q=(x-1/4)^2+y^2
            BigFixed xq = sub(cx, quarter);
            BigFixed q = add(mul(xq, xq), cy2);
            BigFixed left = mul(q, add(q, xq));
            BigFixed right = mul(cy2, quarter);
            return le_signed(left, right);
        }}

        __global__ void mandelbrot_dynamic(
            const unsigned int* x_start_arr, const unsigned int* y_start_arr,
            const unsigned int* x_step_arr,  const unsigned int* y_step_arr,
            int max_iter, int width, int height, int y_offset, int* output
        ) {{
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            int y = blockDim.y * blockIdx.y + threadIdx.y;
            if (x >= width || y >= height) return;

            BigFixed cx_start, cy_start, step_x, step_y;
            for(int i = 0; i < N_LIMBS; i++) {{
                cx_start.limbs[i] = x_start_arr[i]; cy_start.limbs[i] = y_start_arr[i];
                step_x.limbs[i] = x_step_arr[i];    step_y.limbs[i] = y_step_arr[i];
            }}

            unsigned int gy = (unsigned int)(y + y_offset);
            BigFixed cx = add(cx_start, mul_uint(step_x, (unsigned int)x));
            BigFixed cy = add(cy_start, mul_uint(step_y, gy));

            // Start at z=c, equivalent to doing the first Mandelbrot iteration.
            // This saves one full AP iteration for every pixel.
            BigFixed zx = cx;
            BigFixed zy = cy;
            BigFixed zx2 = mul(cx, cx);
            BigFixed zy2 = mul(cy, cy);
            int iter = 1;

            if (in_main_cardioid_or_period2_bulb(cx, cy, zy2)) {{
                output[y * width + x] = max_iter;
                return;
            }}

            while (iter < max_iter) {{
                if (escaped_radius4(zx2, zy2)) break;
                BigFixed zx_zy = mul(zx, zy);
                zy = add(add(zx_zy, zx_zy), cy);
                zx = add(sub(zx2, zy2), cx);
                zx2 = mul(zx, zx);
                zy2 = mul(zy, zy);
                iter++;
            }}
            output[y * width + x] = iter;
        }}
    }}
    '''
    return cp.RawModule(code=cuda_source).get_function("mandelbrot_dynamic")


def required_limbs_for_step(step):
    log2_step = float(step.ln() / LN2)
    return max(4, int(-log2_step // 32) + 5), log2_step


def auto_max_iter(step, args):
    """Adaptive max_iter from pixel scale. Black threshold itself is max_iter."""
    zoom_bits = max(0.0, -float(step.ln() / LN2))
    effective_bits = max(0.0, zoom_bits - args.iter_start_bits)
    value = args.min_iter + args.iter_scale * effective_bits * effective_bits
    value *= args.iter_multiplier
    return int(max(args.min_iter, min(args.max_iter_hard_cap, math.ceil(value))))


# =============================
# Coloring / stats
# =============================

def stats_for_iterations(arr, max_iter):
    capped = arr >= max_iter
    escaped = arr[~capped]
    escaped_nonzero = escaped[escaped > 0]
    capped_count = int(np.count_nonzero(capped))
    return {
        "escaped_count": int(escaped.size),
        "escaped_pct": 100.0 * escaped.size / arr.size if arr.size else 0.0,
        "capped_count": capped_count,
        "capped_pct": 100.0 * capped_count / arr.size if arr.size else 0.0,
        "avg_escaped": float(escaped_nonzero.mean()) if escaped_nonzero.size else 0.0,
        "std_escaped": float(escaped_nonzero.std()) if escaped_nonzero.size else 0.0,
        "max_seen": int(arr.max()) if arr.size else 0,
        "p95": float(np.percentile(escaped_nonzero, 95)) if escaped_nonzero.size else 0.0,
        "p99": float(np.percentile(escaped_nonzero, 99)) if escaped_nonzero.size else 0.0,
    }


def build_gradient_palette_256():
    # 256-color smooth gradient palette.
    stops = np.array([
        [0.00,   0,   7, 100],
        [0.16,  32, 107, 203],
        [0.42, 237, 255, 255],
        [0.64, 255, 170,   0],
        [0.82, 180,   0,   0],
        [1.00, 255, 255, 255],
    ], dtype=np.float32)

    x = stops[:, 0]
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    r = np.interp(t, x, stops[:, 1])
    g = np.interp(t, x, stops[:, 2])
    b = np.interp(t, x, stops[:, 3])

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


PALETTE_256 = build_gradient_palette_256()


def colorize_escape_mod_hwc(arr, max_iter, color_n=1, **_ignored):
    color_n = max(1, int(color_n))

    unknown = arr < 0
    inside = arr >= max_iter

    idx = ((np.maximum(arr, 0).astype(np.int32) // color_n) % 256).astype(np.uint8)
    img = PALETTE_256[idx]

    img[inside | unknown] = 0
    return img

def colorize_histogram_hwc(arr, max_iter):
    h, w = arr.shape
    inside = arr >= max_iter
    escaped = arr[~inside]
    if escaped.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    clipped = np.clip(arr, 0, max_iter)
    hist = np.bincount(np.clip(escaped, 0, max_iter), minlength=max_iter + 1).astype(np.float64)
    hist[0] = 0.0
    cdf = np.cumsum(hist)
    if cdf[-1] > 0:
        cdf /= cdf[-1]

    t = cdf[clipped]
    r = 9.0 * (1.0 - t) * t * t * t * 255.0
    g = 15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0
    b = 8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img[inside] = 0
    return img


# =============================
# Render worker
# =============================

class RenderWorker(threading.Thread):
    def __init__(self, job_queue, result_queue):
        super().__init__(daemon=True)
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.kernel_cache = {}

    def get_kernel(self, n_limbs):
        kernel = self.kernel_cache.get(n_limbs)
        if kernel is None:
            print(f"[CUDA] compiling worker kernel for {n_limbs} limbs ({limbs_to_bits(n_limbs)} bits)")
            kernel = compile_dynamic_kernel(n_limbs)
            self.kernel_cache[n_limbs] = kernel
        return kernel

    def run(self):
        while True:
            job = self.job_queue.get()
            if job is None:
                return
            self.render_job(job)

    def render_job(self, job):
        cancel = job["cancel"]
        job_id = job["job_id"]
        width = job["width"]
        height = job["height"]
        max_iter = job["max_iter"]
        n_limbs = job["n_limbs"]
        chunk_rows = max(1, min(job["chunk_rows"], height))
        x_min = job["x_min"]
        y_min = job["y_min"]
        step_x = job["step_x"]
        step_y = job["step_y"]

        try:
            ensure_decimal_precision(n_limbs)
            kernel = self.get_kernel(n_limbs)
            x_start_dev = cp.array(decimal_to_limbs(x_min, n_limbs), dtype=cp.uint32)
            x_step_dev = cp.array(decimal_to_limbs(step_x, n_limbs), dtype=cp.uint32)
            y_start_dev = cp.array(decimal_to_limbs(y_min, n_limbs), dtype=cp.uint32)
            y_step_dev = cp.array(decimal_to_limbs(step_y, n_limbs), dtype=cp.uint32)
            threads = (16, 16)
            blocks_x = int(np.ceil(width / 16))
            d_out = cp.empty((chunk_rows, width), dtype=cp.int32)

            self.result_queue.put({"type": "started", "job_id": job_id, "height": height})
            start_t = time.time()
            rows_done = 0

            for y0 in range(0, height, chunk_rows):
                if cancel.is_set():
                    self.result_queue.put({"type": "cancelled", "job_id": job_id})
                    return

                h_chunk = min(chunk_rows, height - y0)
                blocks_y = int(np.ceil(h_chunk / 16))

                kernel(
                    (blocks_x, blocks_y),
                    threads,
                    (
                        x_start_dev,
                        y_start_dev,
                        x_step_dev,
                        y_step_dev,
                        max_iter,
                        width,
                        h_chunk,
                        y0,
                        d_out,
                    ),
                )
                arr = d_out[:h_chunk, :].get()
                if cancel.is_set():
                    self.result_queue.put({"type": "cancelled", "job_id": job_id})
                    return

                rows_done += h_chunk
                self.result_queue.put({
                    "type": "chunk",
                    "job_id": job_id,
                    "y0": y0,
                    "arr": arr,
                    "rows_done": rows_done,
                    "height": height,
                })

            elapsed = time.time() - start_t
            self.result_queue.put({"type": "done", "job_id": job_id, "elapsed": elapsed})
        except Exception as e:
            self.result_queue.put({"type": "error", "job_id": job_id, "error": repr(e)})


# =============================
# Small immediate-mode GUI controls
# =============================

class SliderTextControl:
    def __init__(self, name, value, min_value, max_value, step=1, integer=True):
        self.name = name
        self.value = int(value) if integer else float(value)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.integer = integer
        self.dragging = False
        self.active_text = False
        self.text = str(self.value)
        self.slider_rect = pygame.Rect(0, 0, 1, 1)
        self.box_rect = pygame.Rect(0, 0, 1, 1)

    def set_range(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max(min_value, max_value)
        if self.value < self.min_value:
            self.set_value(self.min_value)
        if self.value > self.max_value:
            self.max_value = self.value

    def set_value(self, value):
        value = max(self.min_value, min(self.max_value, value))
        if self.integer:
            value = int(round(value / self.step) * self.step)
        self.value = value
        if not self.active_text:
            self.text = str(self.value)

    def apply_text(self):
        try:
            value = int(float(self.text)) if self.integer else float(self.text)
        except ValueError:
            self.text = str(self.value)
            return False
        if value > self.max_value:
            self.max_value = value
        self.set_value(value)
        self.text = str(self.value)
        return True

    def value_from_x(self, x):
        if self.slider_rect.width <= 0:
            return self.value
        t = (x - self.slider_rect.left) / self.slider_rect.width
        t = max(0.0, min(1.0, t))
        return self.min_value + t * (self.max_value - self.min_value)

    def handle_event(self, event):
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.slider_rect.collidepoint(event.pos):
                self.dragging = True
                self.active_text = False
                self.set_value(self.value_from_x(event.pos[0]))
                changed = True
            elif self.box_rect.collidepoint(event.pos):
                self.active_text = True
                self.text = str(self.value)
            else:
                if self.active_text:
                    changed = self.apply_text()
                self.active_text = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.set_value(self.value_from_x(event.pos[0]))
            changed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = False
                self.set_value(self.value_from_x(event.pos[0]))
                changed = True
        elif event.type == pygame.KEYDOWN and self.active_text:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                changed = self.apply_text()
                self.active_text = False
            elif event.key == pygame.K_ESCAPE:
                self.text = str(self.value)
                self.active_text = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                ch = event.unicode
                if ch and (ch.isdigit() or ch in ".-"):
                    self.text += ch
        return changed

    def draw(self, screen, font, x, y, width):
        label_w = 92
        box_w = 110
        h = 24
        gap = 10
        slider_w = max(100, width - label_w - box_w - 2 * gap)

        label_rect = pygame.Rect(x, y, label_w, h)
        self.slider_rect = pygame.Rect(x + label_w + gap, y + 6, slider_w, 12)
        self.box_rect = pygame.Rect(self.slider_rect.right + gap, y, box_w, h)

        label = font.render(self.name, True, (235, 235, 235))
        screen.blit(label, (label_rect.x, label_rect.y + 3))

        pygame.draw.rect(screen, (60, 60, 60), self.slider_rect, border_radius=4)
        pygame.draw.rect(screen, (150, 150, 150), self.slider_rect, 1, border_radius=4)

        if self.max_value > self.min_value:
            t = (self.value - self.min_value) / (self.max_value - self.min_value)
        else:
            t = 0.0
        t = max(0.0, min(1.0, t))
        knob_x = int(self.slider_rect.left + t * self.slider_rect.width)
        pygame.draw.circle(screen, (230, 230, 230), (knob_x, self.slider_rect.centery), 8)

        box_color = (45, 45, 45) if not self.active_text else (70, 70, 70)
        pygame.draw.rect(screen, box_color, self.box_rect)
        pygame.draw.rect(screen, (220, 220, 220), self.box_rect, 1)
        text_surf = font.render(self.text, True, (255, 255, 255))
        screen.blit(text_surf, (self.box_rect.x + 5, self.box_rect.y + 4))

        range_text = font.render(f"[{self.min_value}..{self.max_value}]", True, (150, 150, 150))
        screen.blit(range_text, (self.box_rect.right + 8, y + 4))


def point_in_controls(pos, render_h):
    return pos[1] >= render_h


def drain_queue(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


# =============================
# Args / main
# =============================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=600, help="Fractal render height; GUI panel is added below it")
    p.add_argument("--min-iter", type=int, default=200)
    p.add_argument("--fixed-max-iter", type=int, default=0, help="If >0, starts with this exact max_iter. GUI can still change it.")
    p.add_argument("--max-iter-hard-cap", type=int, default=10_000_000)
    p.add_argument("--iter-scale", type=float, default=16.0, help="Bigger = more iterations as zoom deepens")
    p.add_argument("--iter-start-bits", type=float, default=8.0, help="No quadratic iteration growth before this pixel-scale zoom depth")
    p.add_argument("--iter-multiplier", type=float, default=1.0)
    p.add_argument("--initial-precision-bits", type=int, default=128)
    p.add_argument("--precision-hard-cap-bits", type=int, default=4096)
    p.add_argument("--chunk-rows", type=int, default=128, help="GPU row strip size. Smaller = more responsive, slower total render.")
    p.add_argument("--color-mode", choices=["histogram", "mod"], default="mod", help="Incremental rendering uses mod live; histogram is applied once complete.")
    return p.parse_args()


def main():
    args = parse_args()
    pygame.init()

    render_w, render_h = args.width, args.height
    panel_h = 128
    screen = pygame.display.set_mode((render_w, render_h + panel_h), pygame.RESIZABLE)
    pygame.display.set_caption("Arbitrary Precision Mandelbrot - streaming chunks")
    font = pygame.font.SysFont("Arial", 16)
    clock = pygame.time.Clock()

    x_min = decimal.Decimal("-2.5")
    x_max = decimal.Decimal("1.0")
    y_min = decimal.Decimal("-1.125")
    y_max = decimal.Decimal("1.125")

    start_iter = args.fixed_max_iter if args.fixed_max_iter > 0 else args.min_iter
    max_iter_control = SliderTextControl("max_iter", start_iter, args.min_iter, max(1000, start_iter * 4), step=1)
    precision_control = SliderTextControl("precision", max(64, args.initial_precision_bits), 64, max(512, args.initial_precision_bits * 4), step=32)
    color_n_control = SliderTextControl("color_N", 4, 1, 256, step=1)

    history = []
    dragging = False
    start_pos = (0, 0)
    end_pos = (0, 0)
    last_stats = None
    last_palette_lo = None
    last_palette_hi = None
    live_palette_lo = None
    live_palette_hi = None
    iter_multiplier = args.iter_multiplier
    auto_iter_enabled = args.fixed_max_iter <= 0
    last_required_bits = bits_to_limbs(precision_control.value) * 32
    last_log2_step = 0.0

    job_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue()
    worker = RenderWorker(job_queue, result_queue)
    worker.start()

    current_cancel = None
    current_job_id = 0
    current_rows_done = 0
    current_rendering = False
    current_error = None
    current_arr = None
    display_rgb = None
    render_surface = pygame.Surface((render_w, render_h))
    render_surface.fill((0, 0, 0))

    def recreate_surface_black():
        nonlocal display_rgb, render_surface, current_arr
        display_rgb = np.zeros((render_h, render_w, 3), dtype=np.uint8)
        current_arr = np.full((render_h, render_w), -1, dtype=np.int32)
        render_surface = pygame.Surface((render_w, render_h))
        render_surface.fill((0, 0, 0))

    def blit_rows_to_surface(y0, y1):
        if display_rgb is None or y1 <= y0:
            return
        sub = render_surface.subsurface((0, y0, render_w, y1 - y0))
        pygame.surfarray.blit_array(sub, np.transpose(display_rgb[y0:y1, :, :], (1, 0, 2)))

    def blit_full_buffer_to_surface():
        pygame.surfarray.blit_array(render_surface, np.transpose(display_rgb, (1, 0, 2)))

    def recolor_current_buffer():
        nonlocal display_rgb
        if current_arr is None or display_rgb is None:
            return
        display_rgb[:, :, :] = colorize_escape_mod_hwc(
            current_arr,
            int(max_iter_control.value),
            color_n=int(color_n_control.value),
        )
        blit_full_buffer_to_surface()

    def start_render(reason):
        nonlocal current_cancel, current_job_id, current_rows_done, current_rendering, current_error
        nonlocal last_required_bits, last_log2_step

        if current_cancel is not None:
            current_cancel.set()
        drain_queue(job_queue)
        drain_queue(result_queue)

        step_x = (x_max - x_min) / decimal.Decimal(render_w)
        step_y = (y_max - y_min) / decimal.Decimal(render_h)
        min_step = min(abs(step_x), abs(step_y))
        required_limbs, last_log2_step = required_limbs_for_step(min_step)
        last_required_bits = limbs_to_bits(required_limbs)

        auto_iter_value = auto_max_iter(min_step, args)
        if auto_iter_enabled:
            if auto_iter_value > max_iter_control.max_value:
                max_iter_control.set_range(args.min_iter, min(args.max_iter_hard_cap, max(auto_iter_value * 2, max_iter_control.max_value)))
            max_iter_control.set_value(auto_iter_value)

        max_iter_control.set_range(
            args.min_iter,
            min(args.max_iter_hard_cap, max(max_iter_control.max_value, max_iter_control.value * 2, auto_iter_value * 2, 1000)),
        )
        precision_control.set_range(
            64,
            min(args.precision_hard_cap_bits, max(precision_control.max_value, precision_control.value * 2, last_required_bits * 2, 512)),
        )
        if precision_control.value < last_required_bits:
            print(f"[PRECISION] auto-raising GUI precision: {precision_control.value} -> {last_required_bits} bits")
            precision_control.set_value(last_required_bits)

        n_limbs = bits_to_limbs(precision_control.value)
        ensure_decimal_precision(n_limbs)
        max_iter = int(max_iter_control.value)

        recreate_surface_black()
        current_job_id += 1
        current_cancel = threading.Event()
        current_rows_done = 0
        current_rendering = True
        current_error = None

        print(f"[RENDER_START] job={current_job_id}, reason={reason}")
        print(f"  view x=[{short_dec(x_min)}, {short_dec(x_max)}]")
        print(f"  view y=[{short_dec(y_min)}, {short_dec(y_max)}]")
        print(f"  step_x={short_dec(step_x)}, step_y={short_dec(step_y)}, log2_step={last_log2_step:.2f}")
        print(f"  cuda_limbs={n_limbs}, cuda_bits={limbs_to_bits(n_limbs)}, required_bits={last_required_bits}, python_decimal_digits={decimal.getcontext().prec}")
        print(f"  max_iter={max_iter} (black threshold is exactly max_iter), chunk_rows={args.chunk_rows}, auto_iter={auto_iter_enabled}")

        job_queue.put({
            "job_id": current_job_id,
            "cancel": current_cancel,
            "width": render_w,
            "height": render_h,
            "x_min": x_min,
            "y_min": y_min,
            "step_x": step_x,
            "step_y": step_y,
            "max_iter": max_iter,
            "n_limbs": n_limbs,
            "chunk_rows": args.chunk_rows,
        })

    recreate_surface_black()
    start_render("initial")

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if current_cancel is not None:
                    current_cancel.set()
                try:
                    job_queue.put_nowait(None)
                except queue.Full:
                    pass
                pygame.quit()
                sys.exit()

            if event.type == pygame.VIDEORESIZE:
                render_w = max(320, event.w)
                render_h = max(240, event.h - panel_h)
                screen = pygame.display.set_mode((render_w, render_h + panel_h), pygame.RESIZABLE)
                start_render("resize")
                continue

            recompute_changed = False
            color_changed = False
            if max_iter_control.handle_event(event):
                auto_iter_enabled = False
                recompute_changed = True
            if precision_control.handle_event(event):
                recompute_changed = True
            if color_n_control.handle_event(event):
                color_changed = True

            if recompute_changed:
                start_render("control")
                continue
            if color_changed:
                recolor_current_buffer()
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    iter_multiplier *= 1.25
                    args.iter_multiplier = iter_multiplier
                    auto_iter_enabled = True
                    print(f"[ITER] multiplier -> {iter_multiplier:.4g}")
                    start_render("iter multiplier up")
                elif event.key == pygame.K_DOWN:
                    iter_multiplier /= 1.25
                    args.iter_multiplier = iter_multiplier
                    auto_iter_enabled = True
                    print(f"[ITER] multiplier -> {iter_multiplier:.4g}")
                    start_render("iter multiplier down")
                elif event.key == pygame.K_a:
                    auto_iter_enabled = not auto_iter_enabled
                    print(f"[ITER] auto max_iter -> {auto_iter_enabled}")
                    start_render("auto toggle")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not point_in_controls(event.pos, render_h):
                dragging = True
                start_pos = event.pos
                end_pos = event.pos

            elif event.type == pygame.MOUSEMOTION and dragging:
                end_pos = (max(0, min(render_w - 1, event.pos[0])), max(0, min(render_h - 1, event.pos[1])))

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and dragging:
                dragging = False
                end_pos = (max(0, min(render_w - 1, event.pos[0])), max(0, min(render_h - 1, event.pos[1])))
                x1, x2 = sorted([start_pos[0], end_pos[0]])
                y1, y2 = sorted([start_pos[1], end_pos[1]])

                if x2 - x1 > 10 and y2 - y1 > 10:
                    history.append((x_min, x_max, y_min, y_max, max_iter_control.value,
                                    precision_control.value, iter_multiplier, auto_iter_enabled))

                    old_x_min, old_x_max = x_min, x_max
                    old_y_min, old_y_max = y_min, y_max
                    dx = old_x_max - old_x_min
                    dy = old_y_max - old_y_min

                    fx1 = decimal.Decimal(x1) / decimal.Decimal(render_w)
                    fx2 = decimal.Decimal(x2) / decimal.Decimal(render_w)
                    fy1 = decimal.Decimal(y1) / decimal.Decimal(render_h)
                    fy2 = decimal.Decimal(y2) / decimal.Decimal(render_h)

                    x_min = old_x_min + fx1 * dx
                    x_max = old_x_min + fx2 * dx
                    y_min = old_y_min + fy1 * dy
                    y_max = old_y_min + fy2 * dy

                    print("[RECT]")
                    print(f"  pixels: x={x1}:{x2}, y={y1}:{y2}, size={x2 - x1}x{y2 - y1}")
                    print(f"  new x=[{short_dec(x_min)}, {short_dec(x_max)}]")
                    print(f"  new y=[{short_dec(y_min)}, {short_dec(y_max)}]")
                    start_render("rectangle")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3 and not point_in_controls(event.pos, render_h):
                if history:
                    (x_min, x_max, y_min, y_max, old_max_iter, old_precision_bits,
                     iter_multiplier, auto_iter_enabled) = history.pop()
                    args.iter_multiplier = iter_multiplier
                    max_iter_control.set_value(old_max_iter)
                    precision_control.set_value(old_precision_bits)
                    print("[UNDO]")
                    print(f"  x=[{short_dec(x_min)}, {short_dec(x_max)}]")
                    print(f"  y=[{short_dec(y_min)}, {short_dec(y_max)}]")
                    start_render("undo")

        # Consume several completed chunks per UI frame.
        # GPU still computes chunks; this only controls UI streaming speed.
        got_chunk = False
        processed_chunks_this_frame = 0
        ui_chunks_per_frame = 8
        while True:
            try:
                msg = result_queue.get_nowait()
            except queue.Empty:
                break

            if msg.get("job_id") != current_job_id:
                continue

            mtype = msg["type"]
            if mtype == "chunk":
                y0 = msg["y0"]
                arr = msg["arr"]
                y1 = y0 + arr.shape[0]

                current_arr[y0:y1, :] = arr
                display_rgb[y0:y1, :, :] = colorize_escape_mod_hwc(
                    arr,
                    int(max_iter_control.value),
                    color_n=int(color_n_control.value),
                )

                current_rows_done = msg["rows_done"]
                blit_rows_to_surface(y0, y1)
                got_chunk = True
                processed_chunks_this_frame += 1
                if processed_chunks_this_frame >= ui_chunks_per_frame:
                    break

            elif mtype == "done":
                current_rendering = False

                if current_arr is not None and np.all(current_arr >= 0):
                    display_rgb[:, :, :] = colorize_escape_mod_hwc(
                        current_arr,
                        int(max_iter_control.value),
                        color_n=int(color_n_control.value),
                    )
                    blit_full_buffer_to_surface()
                    got_chunk = True

                    last_stats = stats_for_iterations(current_arr, int(max_iter_control.value))
                    print("[RENDER_DONE]")
                    print(f"  job={current_job_id}, elapsed={msg['elapsed']:.3f}s")
                    print(f"  escaped_pixels={last_stats['escaped_count']} ({last_stats['escaped_pct']:.2f}%), capped_black_pixels={last_stats['capped_count']} ({last_stats['capped_pct']:.2f}%)")
                    print(f"  avg_escaped_iter={last_stats['avg_escaped']:.3f}, sigma={last_stats['std_escaped']:.3f}, p95={last_stats['p95']:.1f}, p99={last_stats['p99']:.1f}, max_iter_seen={last_stats['max_seen']}")
                    print(f"  color_N={int(color_n_control.value)} -> palette_index=(iter//N)%256")

            elif mtype == "cancelled":
                if msg.get("job_id") == current_job_id:
                    current_rendering = False
            elif mtype == "error":
                current_rendering = False
                current_error = msg["error"]
                print(f"[RENDER_ERROR] job={current_job_id}: {current_error}")

        # Chunks are blitted to their exact row strips when consumed.
        screen.blit(render_surface, (0, 0))

        panel_y = render_h
        pygame.draw.rect(screen, (25, 25, 25), (0, panel_y, render_w, panel_h))
        pygame.draw.line(screen, (110, 110, 110), (0, panel_y), (render_w, panel_y))

        max_iter_control.draw(screen, font, 12, panel_y + 10, render_w - 190)
        precision_control.draw(screen, font, 12, panel_y + 42, render_w - 190)
        color_n_control.draw(screen, font, 12, panel_y + 74, render_w - 190)

        if current_rendering:
            progress_pct = 100.0 * current_rows_done / max(1, render_h)
            stat_text = f"rendering job={current_job_id} rows={current_rows_done}/{render_h} ({progress_pct:.1f}%)"
        elif current_error:
            stat_text = f"error: {current_error}"
        elif last_stats is not None:
            stat_text = f"done | black={last_stats['capped_pct']:.1f}% avg={last_stats['avg_escaped']:.0f} p99={last_stats['p99']:.0f}"
        else:
            stat_text = "ready"

        help_text = (
            f"black threshold=max_iter | bits={limbs_to_bits(bits_to_limbs(precision_control.value))} req={last_required_bits} "
            f"| color_N={color_n_control.value} | auto_iter={'on' if auto_iter_enabled else 'off'} | A toggles auto | drag=zoom | right=back"
        )
        screen.blit(font.render(stat_text, True, (220, 220, 220)), (12, panel_y + 102))
        screen.blit(font.render(help_text, True, (170, 170, 170)), (max(12, render_w - 650), panel_y + 102))

        if dragging:
            rect = pygame.Rect(start_pos[0], start_pos[1], end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            rect.normalize()
            pygame.draw.rect(screen, (255, 255, 255), rect, 2)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
