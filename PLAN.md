# AI Post-Processor - Implementation Plan

## Project Overview

A node-based, style-preserving image enhancement tool that transforms architectural/3D renders into various artistic styles while preserving structural integrity.

**Target Styles (Priority Order):**
1. Watercolor
2. Lithograph
3. Anime/Manga

**Control Inputs:**
- Depth maps
- Normal maps
- Edge/Sketch data
- Segmentation masks

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CUSTOM UI (Your Design)                      │
│                      Intuitive, User-Defined Workflows              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ WebSocket / REST API
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW ENGINE                             │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │  Load Node  │─▶│ Control Node│─▶│  Style Node │─▶│ Blend Node│ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │ Image Node  │─▶│ Parameter   │─▶│  Generate   │─▶│   POST    │─┐│
│  │             │  │   Node      │  │   Node      │  │  PROCESS  │ ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ ││
│                                                                     ││
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ ││
│  │             │  │             │  │             │  │           │ ││
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────┬───┘ ││
│                                                         │     │     │
│                                                         │     ▼     │
│                                                         │  ┌────────┴─┐│
│                                                         │  │   Save   ││
│                                                         │  │   Node   ││
│                                                         │  └──────────┘│
└─────────────────────────────────────────────────────────┴─────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DIFFUSERS BACKEND                              │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │ Style Models    │  │ Control Models  │  │ Adapter Models  │   │
│  │ (Watercolor,    │  │ (Depth, Normal, │  │ (IP-Adapter,    │   │
│  │  Lithograph,    │  │  Canny, Seg)    │  │  Reference)     │   │
│  │  Anime)         │  │                 │  │                 │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Selection

### Base Diffusion Models

| Style | Model | Type | Source |
|-------|-------|------|--------|
| **Watercolor** | Anything V5 / SDXL | Diffusion | Civitai |
| **Lithograph** | Counterfeit-V3 | SD 1.5 Thick Paint | Civitai |
| **Anime/Manga** | DarkSushiMix / MeinaMix | SD 1.5 Anime | Civitai |

### Control Models (All Styles)

| Control Type | Model ID | Purpose |
|--------------|----------|---------|
| Depth | `lllyasviel/control_v11f1p_sd15_depth` | 3D structure preservation |
| Normal | `lllyasviel/control_v11p_sd15_normalbae` | Surface direction |
| Canny | `lllyasviel/control_v11p_sd15_canny` | Edge preservation |
| Scribble | `lllyasviel/control_v11p_sd15_scribble` | Sketch guidance |
| Segmentation | `lllyasviel/control_v11p_sd15_seg` | Semantic masking |

### Style/Reference Models

| Model Type | Model ID | Purpose |
|------------|----------|---------|
| IP-Adapter | `ip-adapter_sd15` | Style reference from images |
| IP-Adapter Plus | `ip-adapter_sd15_plus` | Enhanced style transfer |
| T2I-Adapter | `T2I-Adapter` | Sketch/color/style control |

---

## Post-Processing Pipeline

### Post-Processing Categories

#### 1. **Quality Enhancement**
| Operation | Purpose | Library |
|-----------|---------|---------|
| Sharpening | Enhance edges, reduce blur | OpenCV, PIL |
| Denoising | Remove AI generation artifacts | OpenCV, scikit-image |
| Detail Enhancement | Bring back fine details | OpenCV CLAHE |
| Edge Refinement | Clean up line work | Canny, Sobel filters |

#### 2. **Color & Tone**
| Operation | Purpose | Library |
|-----------|---------|---------|
| Color Correction | Fix AI color shifts | OpenCV, rawpy |
| White Balance | Correct temperature | OpenCV |
| Contrast/Brightness | Final tonal adjustment | OpenCV, PIL |
| Gamma Correction | Fix midtones | OpenCV |
| Saturation Adjustment | Style-dependent color boost | OpenCV |

#### 3. **Style-Specific Post**
| Style | Post-Processing Steps |
|-------|----------------------|
| **Watercolor** | Paper texture overlay, edge softening, color bleeding effect |
| **Lithograph** | High contrast pass, halftone simulation, grain addition |
| **Anime** | Flat color enforcement, cel-edge darkening, screentone patterns |

#### 4. **Resolution & Format**
| Operation | Purpose | Library |
|-----------|---------|---------|
| Upscaling | Increase output resolution | Real-ESRGAN, SwinIR |
| Downscaling | Create preview versions | OpenCV |
| Format Conversion | PNG/JPEG/WebP/etc | PIL |
| Metadata | Embed EXIF/IPTC data | piexif |

#### 5. **Compositing**
| Operation | Purpose | Library |
|-----------|---------|---------|
| Layer Blending | Combine multiple passes | OpenCV, PIL |
| Mask Application | Apply segmentation masks | OpenCV |
| Edge Alpha Blend | Smooth mask boundaries | Gaussian blending |
| Background Replacement | Swap sky/environment | OpenCV |

### Post-Processing Nodes

```
┌─────────────────────────────────────────────────────────────────┐
│                    POST-PROCESSING NODES                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  SharpenNode    │  │  DenoiseNode    │  │  ColorNode      │  │
│  │  - amount       │  │  - strength     │  │  - temperature  │  │
│  │  - radius       │  │  - method       │  │  - contrast     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  UpscaleNode    │  │  TextureNode    │  │  CompositeNode  │  │
│  │  - scale        │  │  - paper/canvas │  │  - blend mode   │  │
│  │  - model        │  │  - intensity    │  │  - mask         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  StyleFilter    │  │  EdgeRefine     │  │  OutputNode     │  │
│  │  - watercolor   │  │  - strengthen   │  │  - format       │  │
│  │  - lithograph   │  │  - smooth       │  │  - quality      │  │
│  │  - anime        │  │                 │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Style-Specific Post-Processing Details

#### Watercolor Post-Processing
```python
# Watercolor-specific post-processing
1. Edge Softening
   - Bilateral filter to smooth while preserving edges
   - Gentle blur on hard edges

2. Paper Texture Overlay
   - Overlay watercolor paper texture
   - Blend mode: multiply/overlay
   - Adjustable intensity

3. Color Bleeding
   - Localized color diffusion
   - Wet-edge simulation

4. Pigment Settling
   - Subtle granulation effect
   - Adds authenticity
```

#### Lithograph Post-Processing
```python
# Lithograph-specific post-processing
1. High Contrast Pass
   - Threshold for binary look
   - Adjustable threshold level

2. Halftone Simulation
   - Dot pattern generation
   - Adjustable dot size/frequency

3. Grain Addition
   - Film grain or lithographic grain
   - Adds texture

4. Edge Crispness
   - Unsharp mask for sharp lines
   - Preserve solid blacks
```

#### Anime Post-Processing
```python
# Anime-specific post-processing
1. Flat Color Enforcement
   - Posterize colors (reduce palette)
   - Solid color areas

2. Cel-Edge Darkening
   - Detect edges, make them solid black
   - Consistent line weight

3. Screentone Patterns
   - Optional halftone shading
   - Manga-style gradient dots

4. Banding Reduction
   - Smooth gradients in shadows
   - Dithering if needed
```

### Upscaling Models

| Model | Best For | VRAM |
|-------|----------|------|
| Real-ESRGAN x4 | General upscaling | ~2GB |
| Real-ESRGAN x2 | Gentle 2x upscaling | ~1GB |
| SwinIR | Detail preservation | ~4GB |
| SCUNet | Denoise + upscale | ~2GB |

---

## Node System Design

### Core Node Types

1. **Input Nodes**
   - `LoadImage` - Load base render
   - `LoadDepth` - Load depth map
   - `LoadNormal` - Load normal map
   - `LoadSegmentation` - Load segmentation mask
   - `LoadReference` - Load style reference image
   - `LoadTexture` - Load paper/canvas texture

2. **Model Nodes**
   - `LoadStyleModel` - Select watercolor/lithograph/anime model
   - `LoadControlNet` - Select control type
   - `LoadAdapter` - Load IP-Adapter/T2I
   - `LoadUpscaler` - Select upscaling model

3. **Parameter Nodes**
   - `PromptNode` - Text prompts (positive/negative)
   - `StrengthNode` - Denoising strength
   - `GuidanceNode` - CFG scale
   - `ControlWeightNode` - ControlNet conditioning scale
   - `StepsNode` - Inference steps

4. **Processing Nodes**
   - `GenerateNode` - Run diffusion pipeline
   - `BlendNode` - Blend multiple results
   - `MaskNode` - Apply segmentation mask
   - `ChainNode` - Chain multiple ControlNets

5. **Post-Processing Nodes**
   - `SharpenNode` - Sharpen with amount/radius
   - `DenoiseNode` - Remove noise
   - `ColorCorrectNode` - Color temperature/contrast
   - `UpscaleNode` - Upscale with selected model
   - `TextureOverlayNode` - Add paper/canvas texture
   - `StyleFilterNode` - Style-specific effects
   - `EdgeRefineNode` - Edge enhancement/smoothing
   - `CompositeNode` - Layer blending with mask

6. **Output Nodes**
   - `SaveImage` - Save result
   - `PreviewImage` - Show in UI
   - `BatchProcess` - Process multiple images
   - `ExportLayers` - Export all intermediate layers

### Workflow Structure (JSON)

```json
{
  "workflow_name": "Watercolor Architecture",
  "nodes": [
    {
      "id": "load_base",
      "type": "LoadImage",
      "params": {"path": "input.png"}
    },
    {
      "id": "load_depth",
      "type": "LoadDepth",
      "params": {"path": "depth.png"}
    },
    {
      "id": "load_control",
      "type": "LoadControlNet",
      "params": {"model": "control_v11f1p_sd15_depth"}
    },
    {
      "id": "load_style",
      "type": "LoadStyleModel",
      "params": {"model": "watercolor_v1"}
    },
    {
      "id": "generate",
      "type": "GenerateNode",
      "params": {
        "prompt": "watercolor painting, architectural sketch",
        "negative_prompt": "photorealistic, sharp, detailed",
        "strength": 0.6,
        "control_weight": 0.9,
        "steps": 30
      },
      "inputs": ["load_base", "load_depth", "load_control", "load_style"]
    },
    {
      "id": "sharpen",
      "type": "SharpenNode",
      "params": {"amount": 0.3, "radius": 1.5},
      "inputs": ["generate"]
    },
    {
      "id": "texture",
      "type": "TextureOverlayNode",
      "params": {
        "texture": "watercolor_paper.png",
        "blend_mode": "multiply",
        "intensity": 0.4
      },
      "inputs": ["sharpen"]
    },
    {
      "id": "upscale",
      "type": "UpscaleNode",
      "params": {"scale": 2, "model": "RealESRGAN_x2"},
      "inputs": ["texture"]
    },
    {
      "id": "save",
      "type": "SaveImage",
      "params": {"path": "output.png"},
      "inputs": ["upscale"]
    }
  ]
}
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] Project structure and dependencies
- [ ] Diffusers pipeline wrapper
- [ ] VRAM management utilities
- [ ] Model loading/caching system

### Phase 2: Style Models
- [ ] Watercolor pipeline (SDXL + ControlNet)
- [ ] Lithograph pipeline (Counterfeit-V3 + ControlNet)
- [ ] Anime pipeline (DarkSushiMix + ControlNet)

### Phase 3: Control System
- [ ] Multi-ControlNet support
- [ ] Depth control
- [ ] Normal control
- [ ] Canny/Edge control
- [ ] Segmentation masking

### Phase 4: Post-Processing Engine
- [ ] Basic image operations (sharpen, denoise, color)
- [ ] Style-specific filters (watercolor, lithograph, anime)
- [ ] Texture overlay system
- [ ] Upscaling integration (Real-ESRGAN)
- [ ] Compositing/blending operations

### Phase 5: Workflow Engine
- [ ] Node system architecture
- [ ] Workflow JSON schema
- [ ] Node execution engine
- [ ] Workflow save/load

### Phase 6: User Workflows
- [ ] Watercolor preset workflow (with post-processing)
- [ ] Lithograph preset workflow (with post-processing)
- [ ] Anime preset workflow (with post-processing)
- [ ] Custom workflow builder

### Phase 7: API Layer
- [ ] WebSocket API for UI
- [ ] Progress callbacks
- [ ] Queue management
- [ ] Error handling

---

## Key Technical Decisions

### Backend: Diffusers Pipeline

**Pros:**
- Familiar from existing code
- Well-documented
- Wide model compatibility
- Active development

**Key Components:**
```python
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    IPAdapterMixin
)
```

### Post-Processing Libraries

```python
# Image processing
import cv2  # OpenCV - core operations
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Advanced operations
from skimage import restoration, filters, exposure
import rawpy  # RAW processing if needed

# Upscaling
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch

# Color science
from colour import Color  # Advanced color operations
```

### Memory Optimization

Reuse techniques from `ai_enhancer.py`:
- `enable_xformers_memory_efficient_attention()`
- `enable_attention_slicing()`
- `vae.enable_slicing()` + `vae.enable_tiling()`
- Aggressive VRAM clearing between operations

### Multi-ControlNet Strategy

```python
# Diffusers supports multiple ControlNets
pipe = StableDiffusionControlNetPipeline(
    controlnet=[depth_controlnet, canny_controlnet, ...]
)

# Each with independent weighting
result = pipe(
    prompt=prompt,
    image=base_image,
    control_image=[depth_img, canny_img, ...],
    controlnet_conditioning_scale=[0.9, 0.5, ...]
)
```

---

## Project Structure

```
Ai-Enhancer_PostProcessor/
├── backend/
│   ├── __init__.py
│   ├── models.py           # Model loading/caching
│   ├── pipelines.py        # Diffusers pipeline wrappers
│   ├── controls.py         # ControlNet management
│   ├── workflows.py        # Workflow engine
│   ├── nodes.py            # Node definitions
│   └── memory.py           # VRAM management
├── postprocess/
│   ├── __init__.py
│   ├── core.py             # Basic image operations
│   ├── color.py            # Color correction
│   ├── filters.py          # Sharpen, denoise, etc.
│   ├── style_fx.py         # Style-specific effects
│   ├── upscale.py          # Upscaling with Real-ESRGAN
│   ├── composite.py        # Layer blending
│   └── textures.py         # Texture overlays
├── styles/
│   ├── watercolor.py       # Watercolor style pipeline
│   ├── lithograph.py       # Lithograph style pipeline
│   └── anime.py            # Anime style pipeline
├── api/
│   ├── __init__.py
│   ├── server.py           # WebSocket/REST server
│   └── schemas.py          # Workflow JSON schemas
├── workflows/
│   ├── watercolor_basic.json
│   ├── lithograph_basic.json
│   ├── anime_basic.json
│   └── presets/
│       ├── watercolor_full.json
│       ├── lithograph_full.json
│       └── anime_full.json
├── textures/
│   ├── watercolor_paper.png
│   ├── canvas.png
│   └── grain.png
├── ui/
│   └── (Your custom UI)
├── requirements.txt
└── PLAN.md
```

---

## Post-Processing Workflow Example

### Complete Watercolor Pipeline

```
Input Render
    │
    ▼
┌─────────────────┐
│  AI Generation  │  Watercolor style + ControlNet
│  (Diffusers)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Edge Softening │  Bilateral filter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Color Boost    │  Increase saturation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Paper Texture  │  Multiply blend
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Sharpen        │  Subtle enhancement
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Upscale 2x     │  Real-ESRGAN
└────────┬────────┘
         │
         ▼
    Output Image
```

---

## Next Steps

1. **Research specific model choices** for each style
2. **Prototype single-style pipeline** (start with watercolor)
3. **Design post-processing chain** for watercolor
4. **Design node system** with your preferred UI workflow
5. **Implement workflow engine**
6. **Add remaining styles**

---

## Model Research To-Do

- [ ] Find optimal watercolor-style LoRA/checkpoint
- [ ] Find lithograph-style model or create custom LoRA
- [ ] Verify anime model choices (DarkSushiMix vs alternatives)
- [ ] Research IP-Adapter integration for style consistency
- [ ] Test multi-ControlNet performance
- [ ] Collect watercolor paper textures
- [ ] Test Real-ESRGAN upscaling quality
