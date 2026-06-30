# GPU Arbitrary-Precision Mandelbrot Viewer

Main file: `mandelbrot.py`

Interactive Mandelbrot explorer in a small, self-contained Python file: `mandelbrot.py`.

The precision is user-adjustable, so the viewer is not tied to a fixed `float64` zoom limit. You can keep increasing the precision bits and dive arbitrarily deep, limited only by GPU time, memory, and patience.

It uses a custom CUDA kernel plus custom fixed-point math on the GPU for arbitrary-precision Mandelbrot zooming, instead of relying on normal GPU floating point.

## Features

- **GPU rendering with CuPy**: Mandelbrot iterations run inside a custom CUDA `RawKernel`.
- **Custom arbitrary-precision GPU math**: complex coordinates are stored as signed fixed-point 32-bit limb arrays and operated on directly inside CUDA. The zoom depth is not capped by normal `float64`; increase the precision bits to continue zooming arbitrarily deep, limited only by compute resources.
- **Dynamic CUDA recompilation**: when precision bits / limb count changes, the self-contained CUDA source is rebuilt for the new limb count.
- **Python `Decimal` coordinate control**: viewport coordinates are held in high precision on the CPU and converted to GPU limbs before rendering.
- **Interactive zooming**: drag a rectangle over the image to zoom into that region.
- **Undo navigation**: right-click to return to the previous viewport.
- **Live chunked rendering**: rendering is split into row chunks, so the image appears progressively instead of freezing until the full frame is done.
- **Cancellable render jobs**: changing zoom, max iterations, or precision cancels the old render and starts a new one.
- **UI controls**: sliders and text boxes control `max_iter` and precision bits.
- **Black threshold = `max_iter`**: pixels that do not escape before `max_iter` are drawn black, like standard Mandelbrot rendering.
- **Small and self-contained**: no project framework, no separate CUDA files; the Python script contains the UI, worker thread, CUDA kernel source, and fixed-point math.

## Install

Install Python dependencies:

```bash
pip install numpy pygame
```

Install the CuPy package matching your CUDA version. Examples:

```bash
pip install cupy-cuda12x
```

or:

```bash
pip install cupy-cuda11x
```

## Run

```bash
python3 mandelbrot.py
```

For more responsive progressive drawing, use smaller chunks:

```bash
python3 mandelbrot.py --chunk-rows 8
```

For faster total rendering but less frequent updates:

```bash
python3 mandelbrot.py --chunk-rows 64
```

## Basic controls

- **Left mouse drag on the image**: select a rectangle and zoom into it.
- **Right click**: undo / go back to the previous viewport.
- **Max iteration slider**: changes escape iteration budget.
- **Max iteration textbox**: type exact `max_iter` manually.
- **Precision bits slider**: changes fixed-point precision.
- **Precision textbox**: type precision manually.
- **While rendering**: you can already select a new rectangle or change settings; the previous render is cancelled.

## Important rendering rule

`max_iter` is both:

1. the maximum number of Mandelbrot iterations, and
2. the black threshold.

A pixel is black when it does not escape before `max_iter`:

```text
if escape_iter >= max_iter:
    pixel = black
else:
    pixel = color_by_escape_iter
```

So increasing `max_iter` gives the renderer more time to decide whether deep-boundary pixels really escape.

## Precision model

The app does not rely on normal GPU floating point for coordinates. Instead it uses signed fixed-point numbers stored as 32-bit limbs:

```text
limbs = [integer/sign limb, fractional limb 1, fractional limb 2, ...]
```

The CUDA kernel implements:

- fixed-point addition
- negation / subtraction
- fixed-point multiplication
- multiplication by pixel index
- Mandelbrot iteration

When the requested precision increases, the app recompiles the CUDA kernel with a new `N_LIMBS` constant.

## Threading / real-time rendering

The UI thread and render worker are separated.

Main/UI thread:

- handles mouse events
- handles sliders/text boxes
- draws the current framebuffer
- shows selection rectangle
- starts/cancels render jobs

Render worker:

- receives a render job id
- renders the image in horizontal row chunks
- sends completed chunks back to the UI
- stops early if a newer job replaces it

This prevents the app from locking up during expensive deep zoom renders.

## Render cancellation behavior

Whenever you change the view or parameters:

1. current render job is marked obsolete
2. canvas is cleared to black
3. a new render job starts
4. chunks from old jobs are ignored if they arrive late

This allows fast exploration even when a previous deep render has not finished.

## Recommended workflow

Start with low iteration and moderate precision, then raise precision as you dive deeper:

```text
max_iter: 200-1000
precision: 128-192 bits
```

When zooming deeper:

- increase precision bits if the image starts showing numerical artifacts
- increase `max_iter` if too many boundary pixels stay black too early
- use smaller `chunk_rows` if the UI feels unresponsive

## Notes

- Very high `max_iter` can be slow even on GPU.
- Very high precision creates larger limb arrays and slower fixed-point multiplication.
- Deep zooming is arbitrary-precision: raise the precision bits as needed; extremely deep zooms are practically limited by render time and GPU memory.
