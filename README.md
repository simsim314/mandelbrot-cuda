# GPU Arbitrary-Precision Mandelbrot Viewer

Main desktop file: `mandelbrot.py`

Browser demo: https://simsim314.github.io/mandelbrot-cuda/

Browser file: `index.html`

This project includes two Mandelbrot viewers:

* a desktop Python/CUDA viewer: `mandelbrot.py`
* a no-backend WebGPU browser viewer: `index.html`

The desktop version uses CUDA through CuPy. The browser version uses WebGPU and runs directly in compatible browsers.

Both desktop and mobile/touch interaction are supported.

## Overview

Interactive Mandelbrot explorer with arbitrary-precision fixed-point math.

The precision is adjustable, so the viewer is not tied to a fixed `float64` zoom limit. You can keep increasing the precision bits called limbs and dive arbitrarily deep, limited only by GPU time, memory and patience.

The desktop version uses a custom CUDA kernel with fixed-point math on the GPU with arbitrary-precision Mandelbrot zooming.

The browser version rewrites the same idea as a WebGPU compute shader using WGSL and 32-bit limb arithmetic.

## Features

* **GPU rendering with CuPy**: the desktop version runs Mandelbrot iterations inside a custom CUDA `RawKernel`.
* `index.html` runs directly in compatible browsers using WebGPU compute shaders.
* **Custom arbitrary-precision GPU math**: complex coordinates are stored as signed fixed-point 32-bit limb arrays and operated on directly on the GPU.
* **Not limited by ****`float64`**** zoom depth**: increase precision bits / limbs to continue zooming deeper, limited only by compute resources.
* **Dynamic CUDA recompilation**: in the desktop version, when precision bits / limb count changes, the CUDA source is rebuilt for the new limb count.
* **Dynamic WGSL recompilation**: in the browser version, changing the limb count recompiles the WebGPU shader.
* **Python ****`Decimal`**** coordinate control**: the desktop version keeps viewport coordinates in high precision on the CPU and converts them to GPU limbs before rendering.
* **JavaScript ****`BigInt`**** coordinate conversion**: the browser version converts decimal viewport coordinates to fixed-point limbs before sending them to WebGPU.
* **Interactive zooming**: drag a rectangle over the image to zoom into that region.
* **Desktop and mobile browser support**: mouse rectangle zoom on desktop, finger-drag rectangle zoom on mobile/touch devices.
* **Undo navigation**: return to the previous viewport.
* **Live chunked rendering**: rendering is split into row chunks, so the image appears progressively.
* **Cancellable render jobs**: changing zoom, max iterations, or precision cancels the old render and starts a new one.
* **UI controls**: controls for `max_iter`, precision / limbs, coloring, and render chunk size.
* **Black threshold = ****`max_iter`**: pixels that do not escape before `max_iter` are drawn black, like standard Mandelbrot rendering.
* **Small and self-contained**: no project framework and no separate CUDA files; the Python script contains the UI, worker thread, CUDA kernel source, and fixed-point math.

## Web demo

Open the WebGPU version here:

```text
https://simsim314.github.io/mandelbrot-cuda/
```

The web version is in:

```text
index.html
```

It uses WebGPU and WGSL shaders, so it can run in compatible desktop and mobile browsers.

Browser requirements:

* WebGPU-capable browser
* WebGPU enabled
* compatible GPU/driver
* HTTPS or GitHub Pages deployment

On desktop, drag with the mouse to select a zoom rectangle.

On mobile, drag with a finger to select a zoom rectangle.

## Desktop install

Install Python dependencies:

```bash
pip install numpy pygame
```

Install the CuPy package matching your CUDA version.

For CUDA 12:

```bash
pip install cupy-cuda12x
```

For CUDA 11:

```bash
pip install cupy-cuda11x
```

## Desktop run

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

## Desktop controls

* **Left mouse drag on the image**: select a rectangle and zoom into it.
* **Right click**: undo / go back to the previous viewport.
* **Max iteration slider**: changes escape iteration budget.
* **Max iteration textbox**: type exact `max_iter` manually.
* **Precision bits slider**: changes fixed-point precision.
* **Precision textbox**: type precision manually.
* **While rendering**: you can already select a new rectangle or change settings; the previous render is cancelled.

## Browser controls

* **Mouse drag**: select a rectangle and zoom on desktop.
* **Finger drag**: select a rectangle and zoom on mobile/touch devices.
* **Undo**: return to the previous viewport.
* **Auto limbs/max_iter**: automatically adapts precision and iteration count to zoom depth.
* **max_iter**: maximum Mandelbrot iteration count.
* **limbs**: number of 32-bit fixed-point limbs used by the WebGPU shader.
* **color_N**: color index divisor; palette index is computed from escape iteration.
* **chunk rows**: number of rows rendered per GPU chunk.
* **width / height**: render resolution.

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

The GPU code implements:

* fixed-point addition
* negation / subtraction
* fixed-point multiplication
* multiplication by pixel index
* Mandelbrot iteration

The desktop version compiles CUDA with a fixed `N_LIMBS` value.

The browser version compiles WGSL with a fixed limb-array size for the current selected limb count. When the limb count changes, the WebGPU shader is rebuilt.

## Coloring

The current coloring rule is:

```text
palette_index = (escape_iter // color_N) % 256
```

Pixels that do not escape before `max_iter` are black.

Increasing `color_N` makes color bands wider.

Decreasing `color_N` makes colors change more rapidly.

## Threading / real-time rendering

The desktop UI thread and render worker are separated.

Main/UI thread:

* handles mouse events
* handles sliders/text boxes
* draws the current framebuffer
* shows selection rectangle
* starts/cancels render jobs

Render worker:

* receives a render job id
* renders the image in horizontal row chunks
* sends completed chunks back to the UI
* stops early if a newer job replaces it

This prevents the app from locking up during expensive deep zoom renders.

The browser version similarly renders in row chunks and displays each chunk progressively.

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

* increase precision bits / limbs if the image starts showing numerical artifacts
* increase `max_iter` if too many boundary pixels stay black too early
* use smaller `chunk_rows` if the UI feels unresponsive
* use larger `chunk_rows` for faster total renders with less frequent visual updates

## Notes

* Very high `max_iter` can be slow even on GPU.
* Very high precision creates larger limb arrays and slower fixed-point multiplication.
* Deep zooming is arbitrary-precision: raise the precision bits or limbs as needed.
* Extremely deep zooms are practically limited by render time, GPU memory, and browser/GPU limits.
* The desktop version is CUDA/CuPy-based.
* The browser version is WebGPU/WGSL-based and does not require a backend.
