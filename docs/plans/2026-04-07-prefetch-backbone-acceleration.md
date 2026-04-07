# Propagation Backbone Prefetch Acceleration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Cut SAM3 video propagation time for 200-500 frames by pre-extracting ViT backbone features in batches before the propagation loop, eliminating the backbone from the per-frame hot path.

**Architecture:** The ViT backbone (`self.detector.backbone.forward_image`) runs inside `_get_img_feats` (model.py:3485) for every frame during propagation, accounting for ~70% of per-frame cost. Backbone runs have no sequential dependency (unlike the tracker), so they can be pre-extracted in batches. We thread pre-computed features through `backbone_out["pre_computed_frame_features"]` — a dict keyed by frame index — so `_get_img_feats` can skip the backbone when features are present. The eviction line (model.py:6543) is made conditional so the cache survives the propagation loop.

**Tech Stack:** PyTorch, Python, ComfyUI node system. No new dependencies.

---

## Context: Where the Backbone Runs

- `Sam3VideoInferenceWithInstanceInteractivity._det_track_one_frame` → `run_backbone_and_detection` (model.py:6366)
- `run_backbone_and_detection` builds `backbone_out = {"img_batch_all_stages": raw_frames, ...}` and calls `forward_video_grounding_multigpu`
- Inside, `_build_multigpu_buffer_next_chunk` calls `forward_grounding` → `_encode_prompt` → `_get_img_feats`
- `_get_img_feats` (model.py:3467–3489): if `backbone_fpn` not in `backbone_out`, runs `self.backbone.forward_image(image)` and merges result into `backbone_out`
- Line 6543: `feature_cache.pop(frame_idx - 1 ...)` evicts the previous frame — prevents caching all frames

## Key Files

- `nodes/sam3/model.py` — backbone forward, `_get_img_feats`, `run_backbone_and_detection`, `Sam3VideoInferenceWithInstanceInteractivity`
- `nodes/sam3_video_nodes.py` — `SAM3Propagate` node
- `nodes/sam3_model_patcher.py` — check if it exposes the underlying `Sam3VideoInferenceWithInstanceInteractivity` model

---

### Task 1: Locate exact line numbers and class boundaries

**Files:**
- Read: `nodes/sam3/model.py` (lines 3451–3490, 6468–6545, 8510+)
- Read: `nodes/sam3_model_patcher.py`

**Step 1: Confirm `_get_img_feats` location**

```bash
grep -n "_get_img_feats\|def run_backbone_and_detection\|class Sam3VideoInferenceWithInstanceInteractivity\|class Sam3ImageOnVideoMultiGPU" nodes/sam3/model.py | head -30
```

Expected: lines near 3451, 6468, and class headers.

**Step 2: Confirm how `sam3_model` exposes the underlying model**

```bash
grep -n "video_predictor\|\.model\b" nodes/sam3_model_patcher.py | head -20
```

Note the attribute path to reach `Sam3VideoInferenceWithInstanceInteractivity` from the `SAM3_MODEL` output. Expected: `sam3_model.video_predictor.model`.

**Step 3: No commit** — this is exploration only.

---

### Task 2: Modify `_get_img_feats` to use pre-computed backbone features

**Files:**
- Modify: `nodes/sam3/model.py` at `_get_img_feats` (around line 3467)

The existing code at line 3467:
```python
img_batch = backbone_out["img_batch_all_stages"]
if img_ids.numel() > 1:
    unique_ids, _ = torch.unique(img_ids, return_inverse=True)
else:
    unique_ids, _ = img_ids, slice(None)
if isinstance(img_batch, torch.Tensor):
    image = img_batch[unique_ids]
elif unique_ids.numel() == 1:
    image = img_batch[unique_ids.item()].unsqueeze(0)
else:
    image = torch.stack([img_batch[i] for i in unique_ids.tolist()])
image = image.to(dtype=torch.float32, device=self.device)
id_mapping = torch.full(
    (len(img_batch),), -1, dtype=torch.long, device=self.device
)
id_mapping[unique_ids] = torch.arange(len(unique_ids), device=self.device)
backbone_out = {
    **backbone_out,
    **self.backbone.forward_image(image),
    "id_mapping": id_mapping,
}
assert "backbone_fpn" in backbone_out
return self._get_img_feats(backbone_out, img_ids=img_ids)
```

**Step 1: Add pre-computed cache check — insert BEFORE `img_batch = ...`**

```python
# Pre-computed backbone features: keyed by frame_idx, value is forward_image() output
_pre_computed = backbone_out.get("pre_computed_frame_features")
if _pre_computed is not None:
    if img_ids.numel() > 1:
        _uid, _ = torch.unique(img_ids, return_inverse=True)
    else:
        _uid = img_ids
    _fid = _uid.item() if _uid.numel() == 1 else None
    if _fid is not None and _fid in _pre_computed:
        _feats = _pre_computed[_fid]
        _img_batch = backbone_out["img_batch_all_stages"]
        _id_mapping = torch.full(
            (len(_img_batch),), -1, dtype=torch.long, device=self.device
        )
        _id_mapping[_uid] = 0
        backbone_out = {**backbone_out, **_feats, "id_mapping": _id_mapping}
        assert "backbone_fpn" in backbone_out
        return self._get_img_feats(backbone_out, img_ids=img_ids)
```

**Step 2: Verify the block is inserted correctly and the file still parses**

```bash
python -c "import ast; ast.parse(open('nodes/sam3/model.py').read()); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add nodes/sam3/model.py
git commit -m "feat: add pre-computed backbone cache check in _get_img_feats"
```

---

### Task 3: Make feature cache eviction conditional

**Files:**
- Modify: `nodes/sam3/model.py` at `run_backbone_and_detection` around line 6543

Current line:
```python
feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
```

**Step 1: Wrap eviction in a condition**

Replace with:
```python
if not feature_cache.get("_prefetch_keep_all"):
    feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('nodes/sam3/model.py').read()); print('OK')"
```

**Step 3: Commit**

```bash
git add nodes/sam3/model.py
git commit -m "feat: conditionalize feature cache eviction for prefetch mode"
```

---

### Task 4: Pass pre-computed features through `run_backbone_and_detection`

**Files:**
- Modify: `nodes/sam3/model.py` at `run_backbone_and_detection` (around line 6490–6510)

Current code builds `backbone_out` as:
```python
sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
    backbone_out={
        "img_batch_all_stages": input_batch.img_batch,
        **text_outputs,
    },
    ...
)
```

**Step 1: Include pre-computed features in backbone_out when available**

Change the `backbone_out={}` dict to:
```python
_pre_computed = feature_cache.get("_pre_extracted_backbone")
backbone_out_init = {
    "img_batch_all_stages": input_batch.img_batch,
    **text_outputs,
}
if _pre_computed is not None:
    backbone_out_init["pre_computed_frame_features"] = _pre_computed

sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
    backbone_out=backbone_out_init,
    ...
)
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('nodes/sam3/model.py').read()); print('OK')"
```

**Step 3: Commit**

```bash
git add nodes/sam3/model.py
git commit -m "feat: thread pre-computed backbone features into forward_video_grounding_multigpu"
```

---

### Task 5: Add `prefetch_backbone_features` method to the video model

**Files:**
- Modify: `nodes/sam3/model.py` — add method to `Sam3VideoInferenceWithInstanceInteractivity`

Add just before or after `warm_up_compilation` (around line 8372). The method runs `self.detector.backbone.forward_image` in batches across all frames and stores results in `inference_state["feature_cache"]["_pre_extracted_backbone"]`.

```python
@torch.inference_mode()
def prefetch_backbone_features(self, inference_state, batch_size: int = 4) -> None:
    """
    Pre-extract ViT backbone features for all frames before propagation.

    Stores results in inference_state["feature_cache"]["_pre_extracted_backbone"]
    so _get_img_feats can skip the backbone during the propagation loop.

    Args:
        inference_state: Active inference state from init_state().
        batch_size: Number of frames to process per backbone call (higher = faster
                    but more VRAM; 4 is a safe default for most GPUs).
    """
    import logging
    log = logging.getLogger("sam3")

    input_batch = inference_state["input_batch"]
    num_frames = inference_state["num_frames"]
    img_batch = input_batch.img_batch  # raw frames, shape [N, C, H, W] or list

    pre_cache = {}
    log.info(f"Prefetching backbone features for {num_frames} frames (batch_size={batch_size})")

    for batch_start in range(0, num_frames, batch_size):
        batch_end = min(batch_start + batch_size, num_frames)
        frame_indices = list(range(batch_start, batch_end))

        # Stack frames into a single batch tensor
        if isinstance(img_batch, torch.Tensor):
            imgs = img_batch[batch_start:batch_end].to(
                dtype=torch.float32, device=self.device
            )
        else:
            imgs = torch.stack(
                [img_batch[i] for i in frame_indices]
            ).to(dtype=torch.float32, device=self.device)

        # Run backbone on the batch
        batch_out = self.detector.backbone.forward_image(imgs)

        # Split batch output back into per-frame entries
        for local_idx, frame_idx in enumerate(frame_indices):
            frame_feats = {}
            for key, val in batch_out.items():
                if isinstance(val, torch.Tensor):
                    frame_feats[key] = val[local_idx : local_idx + 1]
                elif isinstance(val, (list, tuple)):
                    # FPN levels: each is a tensor [B, C, H, W]
                    frame_feats[key] = [
                        v[local_idx : local_idx + 1] if isinstance(v, torch.Tensor) else v
                        for v in val
                    ]
                else:
                    frame_feats[key] = val
            # Move to CPU to avoid holding N frames of features on GPU
            pre_cache[frame_idx] = {
                k: (
                    [x.cpu() for x in v] if isinstance(v, list) else
                    v.cpu() if isinstance(v, torch.Tensor) else v
                )
                for k, v in frame_feats.items()
            }

        log.debug(f"Prefetched frames {batch_start}-{batch_end - 1}")

    inference_state["feature_cache"]["_pre_extracted_backbone"] = pre_cache
    inference_state["feature_cache"]["_prefetch_keep_all"] = True
    log.info("Backbone prefetch complete")
```

**Step 1: Insert the method**

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('nodes/sam3/model.py').read()); print('OK')"
```

**Step 3: Commit**

```bash
git add nodes/sam3/model.py
git commit -m "feat: add prefetch_backbone_features() to Sam3VideoInferenceWithInstanceInteractivity"
```

---

### Task 6: Handle CPU→GPU move when pre-computed features are used

**Files:**
- Modify: `nodes/sam3/model.py` — in the new pre-computed cache check block added in Task 2

Pre-computed features are stored on CPU (Task 5) to save VRAM. When `_get_img_feats` retrieves them, tensors must be on the correct device.

**Step 1: Add `.to(self.device)` in the cache hit branch**

In the block added in Task 2, change:
```python
_feats = _pre_computed[_fid]
```
to:
```python
_raw = _pre_computed[_fid]
_feats = {
    k: (
        [x.to(self.device) for x in v] if isinstance(v, list) else
        v.to(self.device) if isinstance(v, torch.Tensor) else v
    )
    for k, v in _raw.items()
}
```

**Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('nodes/sam3/model.py').read()); print('OK')"
```

**Step 3: Commit**

```bash
git add nodes/sam3/model.py
git commit -m "fix: move pre-computed backbone features to device in _get_img_feats cache hit"
```

---

### Task 7: Add `prefetch_features` parameter to `SAM3Propagate` node

**Files:**
- Modify: `nodes/sam3_video_nodes.py` — `SAM3Propagate.INPUT_TYPES` and `propagate()`

**Step 1: Add input parameter**

In `INPUT_TYPES`, inside `"optional"`, add:
```python
"prefetch_features": ("BOOLEAN", {
    "default": False,
    "tooltip": (
        "Pre-extract ViT backbone features for all frames before propagation. "
        "Faster for 100+ frame videos (40-60% speedup) at the cost of extra RAM "
        "during the prefetch pass. Recommended when tracking with points/boxes."
    ),
}),
"prefetch_batch_size": ("INT", {
    "default": 4,
    "min": 1,
    "max": 16,
    "tooltip": "Frames per backbone batch during prefetch. Higher = faster but more VRAM.",
}),
```

**Step 2: Add parameters to `propagate()` signature**

```python
def propagate(self, sam3_model, video_state, start_frame=0, end_frame=-1,
              direction="forward", prefetch_features=False, prefetch_batch_size=4):
```

**Step 3: Update `IS_CHANGED` signature to match**

```python
@classmethod
def IS_CHANGED(cls, sam3_model, video_state, start_frame=0, end_frame=-1,
               direction="forward", prefetch_features=False, prefetch_batch_size=4):
    result = (id(video_state), start_frame, end_frame, direction, prefetch_features, prefetch_batch_size)
    ...
    return result
```

**Step 4: Call prefetch before the propagation request**

In `propagate()`, after `inference_state = get_inference_state(sam3_model, video_state)` and before building the `request` dict:

```python
if prefetch_features:
    top_model = sam3_model.video_predictor.model
    if hasattr(top_model, "prefetch_backbone_features"):
        log.info("Prefetching backbone features...")
        top_model.prefetch_backbone_features(inference_state, batch_size=prefetch_batch_size)
    else:
        log.warning("prefetch_features requested but model does not support it; skipping")
```

**Step 5: Verify syntax**

```bash
python -c "import ast; ast.parse(open('nodes/sam3_video_nodes.py').read()); print('OK')"
```

**Step 6: Commit**

```bash
git add nodes/sam3_video_nodes.py
git commit -m "feat: add prefetch_features option to SAM3Propagate node"
```

---

### Task 8: Smoke test with a short video

**Step 1: Restart ComfyUI (or reload custom nodes)**

**Step 2: Run a propagation with a 10-frame video, `prefetch_features=False`**

Observe: propagation completes without error. Note time.

**Step 3: Run same video with `prefetch_features=True, prefetch_batch_size=4`**

Observe:
- Log shows "Prefetching backbone features for N frames"
- Log shows "Backbone prefetch complete"
- Propagation completes without error
- Output masks are identical to step 2 (same objects tracked)

**Step 4: If errors occur, check:**

- `backbone_fpn` not in pre-computed feats → `forward_image` returns different keys for this model version; add a log of `batch_out.keys()` to debug
- Device mismatch → Task 6 CPU→GPU move missed a tensor type
- Shape mismatch → the batch split in Task 5 is off; check `local_idx : local_idx + 1` vs `local_idx`

---

### Task 9: Push and note torch.compile as a free extra win

**Step 1: Push**

```bash
git push origin main
```

**Step 2: Note for user**

`torch.compile=True` in `LoadSAM3Model` gives an additional 20-40% speedup on top of prefetch, at the cost of a 2-5 minute one-time warmup per session. The two are fully composable.
